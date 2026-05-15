import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation


class TorchLayer(torch.nn.Module):
    @property
    def torch_params(self):
        if not hasattr(self, "_torch_params"):
            self._track_variables()
        return self._torch_params

    def _post_build(self):
        # Do not track variables when in a stateless scope.
        # The variables are not initialized.
        if in_stateless_scope():
            return
        self._track_variables()

    def _track_variables(self):
        # set torch_params attribute will have module automatically track
        # parameters.
        self._torch_params = torch.nn.ParameterDict(
            {variable.path: variable.value for variable in self.variables}
        )

    def named_parameters(
        self,
        prefix="",
        recurse=True,
        remove_duplicate=True,
    ):
        if not hasattr(self, "_torch_params"):
            self._track_variables()
        return torch.nn.Module.named_parameters(
            self, prefix, recurse, remove_duplicate
        )

    def forward(self, *args, **kwargs):
        return Operation.__call__(self, *args, **kwargs)

    def _setattr_hook(self, name, value):
        # Skip the wrapping logic for non-Module values so `torch._dynamo`
        # does not specialize on every distinct `name` string and blow
        # past `recompile_limit` (default 8).
        if not isinstance(value, torch.nn.Module):
            return name, value

        from keras.src.layers import Layer
        from keras.src.utils.torch_utils import TorchModuleWrapper

        if (
            isinstance(value, Layer)
            or name == "_torch_params"
            or isinstance(self, TorchModuleWrapper)
        ):
            return name, value
        return name, TorchModuleWrapper(value)

    def _post_track_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path not in self.torch_params:
                self.torch_params[variable.path] = variable.value

    def _post_untrack_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path in self.torch_params:
                self.torch_params.pop(variable.path)
