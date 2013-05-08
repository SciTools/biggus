Biggus
======

|build_status|


Virtual large arrays and lazy evaluation.


Design goals:

- Keep the public interface compact.
- Leverage standard Python syntax.
- Avoid overloading behaviour.
- Mimic NumPy when it doesn't contradict the other goals.


Use cases:

1. Extract a lazy subset of a lazy array.

2. Extract a sequence of concrete slices from a lazy array.

   - MUST NOT make the full lazy array concrete.

3. Stack a homogenous collection of lazy arrays to create a higher
   dimensional lazy array.

4. Join a collection of compatible lazy arrays to create a larger
   lazy array of the same dimensionality.


.. |build_status| image:: https://secure.travis-ci.org/SciTools/biggus.png
   :alt: Build Status
   :target: http://travis-ci.org/SciTools/biggus
