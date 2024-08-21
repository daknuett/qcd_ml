qcd_ml Datatypes
================

``qcd_ml`` uses ``torch`` tensors to store an manipulate all matrices and
fields. All data is in double precision unless explicitly marked as half
precision.

- All fields have ``x, y, z, t`` as the first four indices.
- :math:`SU(3)` matrices are ``3,3`` matrices.
- Spin matrices are ``4,4`` matrices.
- Spin-color vectors have indices ``s,g`` where ``s`` is the spin index and
  ``g`` is the gauge index.
- A gauge transformation ``V`` is a :math:`SU(3)` field.
