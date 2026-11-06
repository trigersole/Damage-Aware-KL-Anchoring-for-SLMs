"""Input provenance, immutable resumable blocks and diagnostic neighbor reporting.

Only the numeric helpers import numpy, lazily. Check/report require the stdlib.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import io_utils as io


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_item(row):
    try:
        value = json.loads(row['example_id'])
    except json.JSONDecodeError:
        value = row['example_id']
    if isinstance(value, list):
        require(len(value) == 6, 'Expected original source ID plus one view')
        value = value[:5]
    return json.dumps([row['dataset'], value], ensure_ascii=False, separators=(',', ':'))


def verify_source(folder, config):
    expected = config['source']
    require(io.file_hash(folder / 'identity.json') == expected['identity_sha256'], 'Preflight identity bytes changed')
    require(io.digest(io.read(folder / 'identity.json')) == expected['fingerprint'], 'Preflight fingerprint changed')
    for name, stage in expected['stages'].items():
        require(io.read(folder / name / 'stage.json') == stage, 'Preflight stage receipt changed: ' + name)
        actual = {str(p.relative_to(folder / name)) for p in (folder / name).rglob('*') if p.is_file()}
        require(actual == set(stage['files']) | {'stage.json'}, 'Preflight stage file set changed: ' + name)
        for filename, sha in stage['files'].items():
            require(io.file_hash(folder / name / filename) == sha, 'Preflight file changed: ' + filename)
    manifest = io.read(folder / 'review/review_manifest.json')
    require(manifest['bundle_sha256'] == expected['review_bundle_sha256'], 'Review identity changed')
    require(manifest['semantic_screen_complete'] is False, 'Expected original lexical-only review')
    answers = io.rows(folder / 'review/review_answers.template.jsonl')
    require(all(r['decision'] == 'pending' for r in answers), 'Original formal review must remain pending')


class Corpus:
    def __init__(self, candidates, references, flags, hard_ids=()):
        self.texts, self.lookup = [], {}
        self.candidates, self.references, self.flags = [], [], flags
        self.text_candidates, self.text_references = defaultdict(list), defaultdict(list)
        all_ids = set()
        hard_ids = set(hard_ids)
        for row in candidates:
            require(set(row) == {'example_id', 'prompt'}, 'Unexpected candidate fields')
            require(row['example_id'] not in all_ids, 'Duplicate candidate ID')
            all_ids.add(row['example_id'])
            if row['example_id'] in hard_ids:
                continue
            row = dict(row, text_index=self.intern(row['prompt']))
            self.text_candidates[row['text_index']].append(len(self.candidates))
            self.candidates.append(row)
        require(hard_ids <= all_ids, 'Hard exclusion without candidate')
        reference_ids = set()
        for row in references:
            require(set(row) == {'dataset', 'example_id', 'text'}, 'Unexpected reference fields; inputs only')
            key = (row['dataset'], row['example_id'])
            require(key not in reference_ids, 'Duplicate reference view ID')
            reference_ids.add(key)
            row = dict(row, text_index=self.intern(row['text']), source_item=source_item(row))
            self.text_references[row['text_index']].append(len(self.references))
            self.references.append(row)
        self.candidate_by_id = {r['example_id']: r for r in self.candidates}
        self.reference_by_id = {(r['dataset'], r['example_id']): r for r in self.references}

    def intern(self, text):
        require(isinstance(text, str) and text.strip(), 'Empty/non-string full text')
        if text not in self.lookup:
            self.lookup[text] = len(self.texts)
            self.texts.append(text)
        return self.lookup[text]

    def mapping(self):
        for index, text in enumerate(self.texts):
            yield {'text_index': index, 'utf8_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                   'candidate_ids': [self.candidates[i]['example_id'] for i in self.text_candidates.get(index, [])],
                   'reference_views': [{k: self.references[i][k] for k in ('dataset', 'example_id', 'source_item')}
                                       for i in self.text_references.get(index, [])]}

    def inventory(self):
        view_counts = Counter(r['source_item'] for r in self.references)
        return {'active_candidates': len(self.candidates), 'input_views': len(self.references),
                'unique_byte_identical_texts': len(self.texts), 'max_views_per_source_item': max(view_counts.values(), default=0),
                'input_views_by_dataset': dict(Counter(r['dataset'] for r in self.references)),
                'mapping_fingerprint': io.digest(list(self.mapping()))}


def load_corpus(folder, config):
    verify_source(folder, config)
    r = folder / 'review'
    candidates, refs = io.rows(r / 'candidates.jsonl'), io.rows(r / 'heldout_inputs.jsonl')
    hard, flags = io.rows(r / 'hard_exclusions.jsonl'), io.rows(r / 'review_flags.jsonl')
    expected = config['expected']
    require(len(candidates) == expected['all_candidates'], 'Candidate count changed')
    require(len(hard) == expected['hard_excluded'], 'Hard exclusion count changed')
    require(len(refs) == expected['input_views'], 'Input view count changed')
    counts = Counter(row['kind'] for row in flags)
    require(counts == {'benchmark_overlap': expected['lexical_pairs'], 'spot_check': expected['spot_checks']}, 'Lexical references changed')
    corpus = Corpus(candidates, refs, flags, [x['example_id'] for x in hard])
    require(len(corpus.candidates) == expected['active_candidates'], 'Active candidate count changed')
    require(corpus.inventory()['max_views_per_source_item'] <= config['neighbors']['max_views_per_source_item'], 'Neighbor grouping coverage assumption changed')
    for flag in flags:
        require(flag['candidate_id'] in corpus.candidate_by_id, 'Flag refers to excluded/absent candidate')
        if flag['kind'] == 'benchmark_overlap':
            require((flag['dataset'], flag['heldout_id']) in corpus.reference_by_id, 'Flag reference missing')
    return corpus


def verified_block(path, fingerprint):
    receipt = io.read(path / 'receipt.json')
    require(receipt['fingerprint'] == fingerprint, 'Cached block identity changed')
    actual = {p.name for p in path.iterdir() if p.is_file()}
    require(actual == set(receipt['files']) | {'receipt.json'}, 'Cached block file set changed')
    for name, sha in receipt['files'].items():
        require(io.file_hash(path / name) == sha, 'Cached block corrupt: ' + name)
    return receipt


def block(parent, name, fingerprint, build, budget):
    path = parent / name
    if path.exists():
        return verified_block(path, fingerprint)
    budget.check()
    parent.mkdir(parents=True, exist_ok=True)
    # Only our disposable, never-published scratch paths are removed after interruption.
    for stale in parent.glob('.' + name + '.tmp-*'):
        if stale.is_dir():
            shutil.rmtree(stale)
    temp = parent / ('.' + name + '.tmp-' + uuid.uuid4().hex)
    temp.mkdir()
    try:
        metadata = build(temp) or {}
        receipt = dict(metadata, fingerprint=fingerprint,
                       files={p.name: io.file_hash(p) for p in temp.iterdir() if p.is_file()})
        io.json_file(temp / 'receipt.json', receipt)
        os.rename(temp, path)
        return receipt
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def length_block(texts, start, tokenizers, budget):
    result = []
    budget.check()
    bge = tokenizers[0](texts, add_special_tokens=True, truncation=False, padding=False)['input_ids']
    budget.check()
    llama = tokenizers[1](texts, add_special_tokens=False, truncation=False, padding=False)['input_ids']
    require(len(bge) == len(llama) == len(texts), 'Tokenizer batch count changed')
    for offset, (x, y) in enumerate(zip(bge, llama)):
        result.append({'text_index': start + offset, 'bge_tokens_with_specials': len(x), 'llama_raw_tokens': len(y)})
    return result


def describe(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return dict(count=len(values), min=values[0], max=values[-1],
                **{name: values[round((len(values)-1)*p)] for name, p in [('p50', .5), ('p90', .9), ('p95', .95), ('p99', .99)]})


def length_summary(corpus, lengths, max_tokens):
    require([r['text_index'] for r in lengths] == list(range(len(corpus.texts))), 'Length index coverage changed')
    by_index = {r['text_index']: r for r in lengths}
    result = {'unique_texts': len(lengths), 'encoder_limit_includes_native_special_tokens': max_tokens,
              'bge_unique_texts': describe([r['bge_tokens_with_specials'] for r in lengths]),
              'llama_raw_unique_texts': describe([r['llama_raw_tokens'] for r in lengths]),
              'counts_are_raw_prompts_not_future_teacher_student_sequences': True}
    for name, rows in [('candidates', corpus.candidates), ('input_views', corpus.references)]:
        result[name] = {field: describe([by_index[r['text_index']][field] for r in rows])
                        for field in ('bge_tokens_with_specials', 'llama_raw_tokens')}
        result[name]['llama_raw_above_bins'] = {str(cap): sum(by_index[r['text_index']]['llama_raw_tokens'] > cap for r in rows)
                                                    for cap in [512, 1024, 2048, 4096, 8192]}
    over = [dict(r, candidate_ids=[corpus.candidates[i]['example_id'] for i in corpus.text_candidates.get(r['text_index'], [])],
                 reference_views=[{k: corpus.references[i][k] for k in ('dataset','example_id')}
                                  for i in corpus.text_references.get(r['text_index'], [])])
            for r in lengths if r['bge_tokens_with_specials'] > max_tokens]
    result['over_encoder_limit_unique_texts'] = len(over)
    return result, over


def check_vectors(vectors, count, dimension):
    import numpy as np
    require(vectors.dtype == np.float32 and vectors.shape == (count, dimension), 'Embedding dtype/shape changed')
    require(bool(np.isfinite(vectors).all()), 'Non-finite embedding')
    require(bool(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=3e-5, rtol=0)), 'Embeddings must have unit norm')


def stable_top(scores, indices, k):
    """Exact top-k, with lowest canonical reference index winning an exact tie."""
    import numpy as np
    require(scores.ndim == 2 and scores.shape == indices.shape, 'Top-k array mismatch')
    out_s, out_i = [], []
    for score, index in zip(scores, indices):
        if len(score) > k:
            cut = np.partition(score, len(score)-k)[len(score)-k]
            better = np.flatnonzero(score > cut)
            tied = np.flatnonzero(score == cut)
            tied = tied[np.argsort(index[tied], kind='stable')[:k-len(better)]]
            selected = np.concatenate([better, tied])
        else:
            selected = np.arange(len(score))
        selected = selected[np.lexsort((index[selected], -score[selected]))][:k]
        out_s.append(score[selected]); out_i.append(index[selected])
    return np.asarray(out_s), np.asarray(out_i)


def search(query_indices, reference_indices, vectors, k, chunk, multiply, budget, exclude_self=False):
    import numpy as np
    query = np.asarray(vectors[query_indices], dtype=np.float32)
    best_s = np.empty((len(query), 0), dtype=np.float32)
    best_i = np.empty((len(query), 0), dtype=np.int64)
    for start in range(0, len(reference_indices), chunk):
        budget.check()
        refs = np.asarray(reference_indices[start:start+chunk], dtype=np.int64)
        scores = multiply(query, np.asarray(vectors[refs], dtype=np.float32))
        require(scores.shape == (len(query_indices), len(refs)) and bool(np.isfinite(scores).all()), 'Invalid cosine matrix')
        indices = np.broadcast_to(refs, scores.shape)
        if exclude_self:
            scores[np.asarray(query_indices)[:, None] == indices] = -np.inf
        best_s, best_i = stable_top(np.concatenate([best_s, scores], axis=1),
                                   np.concatenate([best_i, indices], axis=1), min(k, best_s.shape[1]+len(refs)))
    return best_s, best_i


def grouped_references(corpus, scores, text_indices, k):
    groups = {}
    for score, index in zip(scores, text_indices):
        for reference_index in corpus.text_references[int(index)]:
            ref = corpus.references[reference_index]
            group = groups.setdefault(ref['source_item'], {'dataset': ref['dataset'], 'source_item': ref['source_item'],
                                                         'score': float(score), 'best_view_id': ref['example_id'],
                                                         'best_text_index': int(index)})
            if score > group['score']:
                group.update(score=float(score), best_view_id=ref['example_id'], best_text_index=int(index))
    return sorted(groups.values(), key=lambda x: (-x['score'], x['source_item']))[:k]


def candidate_neighbors(corpus, self_id, scores, indices, k):
    result = []
    for score, index in zip(scores, indices):
        for candidate_index in corpus.text_candidates[int(index)]:
            row = corpus.candidates[candidate_index]
            if row['example_id'] != self_id and float(score) != float('-inf'):
                result.append({'candidate_id': row['example_id'], 'score': float(score), 'text_index': int(index)})
    return sorted(result, key=lambda x: (-x['score'], x['candidate_id']))[:k]
