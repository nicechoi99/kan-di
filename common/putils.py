"""putils.py — minimal serial execution helper.

The full study used a Ray-based distributed backend to run many
BO seeds/datasets in parallel across a cluster. That infrastructure code
is intentionally omitted from this release; this lightweight module provides
a drop-in *sequential* replacement so the framework runs on a single machine.

Exports
-------
ray            : the real ``ray`` module if installed, else a small dummy whose
                 ``ray._raylet.ObjectRef`` type never matches ordinary objects.
parallel_eval  : evaluate ``func`` over ``inputs`` sequentially.
"""

try:
    import ray  # optional; only used if a real cluster is available
except Exception:                       # pragma: no cover
    class _DummyRaylet:
        class ObjectRef:                # sentinel type; nothing is ever this type
            pass

    class _DummyRay:
        _raylet = _DummyRaylet

        @staticmethod
        def get(x):
            return x

        @staticmethod
        def put(x):
            return x

    ray = _DummyRay()


def parallel_eval(func, inputs, params=None, parallel=False, resume=False,
                  target='local', debug=False, **kwargs):
    """Sequentially apply ``func(x, params=params, i=i)`` over ``inputs``.

    Drop-in replacement for the original parallel evaluator. ``parallel`` and
    ``target`` are accepted for signature compatibility but ignored — execution
    is always serial in this release.
    """
    results = []
    for i, x in enumerate(inputs):
        results.append(func(x, params=params, i=i))
    return results
