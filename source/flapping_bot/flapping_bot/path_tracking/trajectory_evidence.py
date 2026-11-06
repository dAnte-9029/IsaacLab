"""Read-only v2 physical evidence. Exact conditions, trajectory and contract only."""
import json
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def identity(spec, plant, initial, contract):
    return canonical(dict(trajectory=spec.to_dict(), plant=plant, initial=initial, contract=contract))


class OfflineEvidence:
    """Private immutable JSON snapshot; callers receive copies. Never writes source."""
    def __init__(self, filename):
        self._text = Path(filename).read_text()
        rules = json.loads(self._text)
        if rules['schema_version'] != 2:
            raise ValueError('Requires v2 evidence')
        self.count = len(rules['evidence'])

    @property
    def records(self):
        return json.loads(self._text)['evidence']

    def query(self, spec, backend, conditions, contract):
        matches = [r for r in self.records if r['backend'] == backend
                   and r['physical_conditions'] == conditions and r['trajectory'] == spec.to_dict()]
        success = [r for r in matches if r['status'] == 'VERIFIED FEASIBLE']
        exact = [r for r in success if r['evaluation_contract'] == contract]
        status = ('VERIFIED FEASIBLE' if exact else 'RELATED EVIDENCE / TARGET UNVERIFIED' if success
                  else 'NOT FOUND' if any(r['evaluation_contract'] == contract for r in matches)
                  else 'UNTESTED / UNCERTAIN')
        return dict(status=status, witnesses=exact, related=success, hard_exclude=False)
