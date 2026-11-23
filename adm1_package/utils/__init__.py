"""Shared utilities.

======================  =====================================================
``parameters``          mechanistic parameter container and JSON persistence
``datasets``            file order, measurements, input scaling, batching
``forcing``             how the daily records are interpolated in time
``augmentation``        noise augmentation
``losses``              residual scaling for the multi-objective loss
``nn_architectures``    the correction network and its input vector
``optimizers``          optimizer, learning-rate and curriculum schedules
``solver``              ODE solver and tolerances, shared by every variant
``ude_trainer``         the loss, training loop and restore of the hybrid models
``checkpoint``          what a trained fold stores, and how it is read back
``initial_state``       calibrated or warmed-up initial states
``gas``                 biogas and methane outflow from the head-space balance
``corrections``         how network outputs become model parameters
``solids``              total and volatile solids from the model states
``plotting``            names and units of the outputs, series for figures
``plot_style``          figure sizes, colours and rcParams
======================  =====================================================

Import the submodules themselves (``from adm1_package.utils.gas import ...``); this
package imports none of them, so reading one does not load the trainer, Orbax or
PyTorch.
"""
