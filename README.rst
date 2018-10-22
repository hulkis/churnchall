===============================
churnchall
===============================

.. image:: https://travis-ci.org/hulkis/churnchall.svg?branch=master
    :target: https://travis-ci.org/hulkis/churnchall
.. image:: https://readthedocs.org/projects/churnchall/badge/?version=latest
   :target: http://churnchall.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://coveralls.io/repos/github/hulkis/churnchall/badge.svg
   :target: https://coveralls.io/github/hulkis/churnchall
.. image:: https://badge.fury.io/py/churnchall.svg
   :target: https://pypi.python.org/pypi/churnchall/
   :alt: Pypi package


DataScience Challenge gas and power supply

:License: MIT license
:Documentation: http://churnchall.readthedocs/en/latest
:Source: https://github.com/hulkis/churnchall


Installation
------------

.. code:: bash

    pip install churnchall


Results History
---------------

- On tag v0.0.1:

  No features done, nada, except some lgb parameters tuning (very few)

  .. code:: bash

    churnchall lgb validate --debug=False --num-boost-round=10000 --early-stopping-rounds=100
    # [1490]  train's auc: 0.997888   train's xentropy: 0.0264245     train's AUC Lift: 0.983196      test's auc: 0.986296    test's xentropy: 0.0427691      test's AUC Lift: 0.972689

    churnchall lgb cv --debug=False --num-boost-round=10000 --early-stopping-rounds=100
    #[862]   cv_agg's auc: 0.984023 + 0.00104785     cv_agg's xentropy: 0.0479363 + 0.00345682       cv_agg's AUC Lift: 0.969892 + 0.00101755

    churnchall lgb generate-submit --debug=False --num-boost-round=1200
    # Public score: 0.8258142519

- On tag v0.0.2:

  No features done, nada, except some lgb parameters tuning (very few)

  .. code:: bash

    churnchall lgb generate-submit --debug=False --num-boost-round=1200
    # Public score: 0.8132866009

- On tag v0.0.3:

  .. code:: bash

    churnchall xgb cv --debug=False --num-boost-round=4000 --early-stopping-rounds=300
    #[784]   train-AUC_Lift:0.985376+9.80804e-05     train-auc:0.999978+2.63818e-06  test-AUC_Lift:0.972193+0.00115528       test-auc:0.986393+0.00106838
    #[785]   train-AUC_Lift:0.985376+9.82291e-05     train-auc:0.999978+2.63818e-06  test-AUC_Lift:0.972194+0.00115615       test-auc:0.986395+0.00106889


References
----------

- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
