"""Offline evaluation of aligned labels. No Qt, database or algorithm mutation."""
import argparse
import hashlib
import json
from pathlib import Path

CLASSES = ('good', 'partial', 'absent', 'undefined')


def _ratio(a, b):
    return a / b if b else None


def _segments(labels):
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            yield start, end, labels[start]
            start = end


def score(truth, prediction):
    """Score one common, uniformly sampled index axis; boundaries are in samples."""
    if not truth or len(truth) != len(prediction):
        raise ValueError('Expected nonempty aligned label arrays')
    if any(type(x) is not int or x not in range(4) for x in [*truth, *prediction]):
        raise ValueError('Labels must be integers 0..3; missing annotation is not class 3')
    confusion = [[0] * 4 for _ in range(4)]
    for actual, predicted in zip(truth, prediction):
        confusion[actual][predicted] += 1
    per_class = {}
    for k, name in enumerate(CLASSES):
        tp = confusion[k][k]
        support = sum(confusion[k])
        selected = sum(row[k] for row in confusion)
        per_class[name] = dict(support=support, precision=_ratio(tp, selected),
                               recall=_ratio(tp, support), f1=_ratio(2 * tp, support + selected),
                               iou=_ratio(tp, support + selected - tp))
    actual_segments, predicted_segments = list(_segments(truth)), list(_segments(prediction))
    def best_overlaps(left, right):
        return [max((max(0, min(b, d) - max(a, c)) /
                     (max(b, d) - min(a, c)) for c, d, other in right if other == label),
                    default=0.) for a, b, label in left]
    def boundaries(labels):
        return [(i, labels[i - 1], labels[i]) for i in range(1, len(labels))
                if labels[i] != labels[i - 1]]
    a, b = boundaries(truth), boundaries(prediction)
    distances, unmatched = [], 0
    for left, right in ((a, b), (b, a)):
        for i, before, after in left:
            candidates = [abs(i - j) for j, x, y in right if (before, after) == (x, y)]
            if candidates:
                distances.append(min(candidates))
            else:
                unmatched += 1
    return dict(confusion=confusion, per_class=per_class,
                accuracy=sum(confusion[k][k] for k in range(4)) / len(truth),
                predicted_undefined_fraction=prediction.count(3) / len(truth),
                expert_undefined_fraction=truth.count(3) / len(truth),
                expert_segment_best_iou=best_overlaps(actual_segments, predicted_segments),
                predicted_segment_best_iou=best_overlaps(predicted_segments, actual_segments),
                boundary_mean_distance_samples=_ratio(sum(distances), len(distances)),
                boundary_unmatched=unmatched,
                expert_boundary_count=len(a), predicted_boundary_count=len(b))


def evaluate(document):
    """Validate well-level split before scoring every held-out study separately."""
    if document.get('schema') != 'vac-evaluation-v1':
        raise ValueError('Unsupported evaluation schema')
    provenance = document['provenance']
    for key in ('algorithm_version', 'code_revision', 'reference_sha256',
                'parameters', 'annotation_revision', 'hypothesis'):
        if key not in provenance or provenance[key] in (None, ''):
            raise ValueError('Missing provenance: ' + key)
    groups, inputs, identities, results = {}, {}, set(), {}
    for study in document['studies']:
        well, identity, split = study['well_id'], study['study_id'], study['split']
        if not isinstance(well, str) or not well or not isinstance(identity, str) or not identity:
            raise ValueError('Nonempty well and study identifiers required')
        if split not in ('train', 'validation', 'evaluation') or identity in identities:
            raise ValueError('Invalid split or duplicate study')
        identities.add(identity)
        if well in groups and groups[well] != split:
            raise ValueError('Well occurs in multiple splits: ' + well)
        groups[well] = split
        digest = study['input_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Expected SHA-256 of input dataset')
        if digest in inputs and inputs[digest] != split:
            raise ValueError('Input dataset occurs in multiple splits')
        inputs[digest] = split
        if study.get('axis') != 'uniform_sample_index':
            raise ValueError('Explicit aligned uniform sample index required')
        if split == 'evaluation':
            results[identity] = score(study['truth'], study['prediction'])
    if not results:
        raise ValueError('No independent evaluation studies')
    return dict(schema='vac-evaluation-result-v1', provenance=provenance,
                classes=list(CLASSES), studies=results,
                quality_claim='Metrics only; expert acceptance required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    result = evaluate(json.loads(payload))
    result['evaluation_input_sha256'] = hashlib.sha256(payload).hexdigest()
    result['evaluator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
