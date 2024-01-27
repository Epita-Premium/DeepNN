import torch


class Grad_Paper:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(self.save_activation(name)))
                self.hooks.append(module.register_backward_hook(self.save_gradient(name)))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output

        return hook

    def save_gradient(self, name):
        def hook(module, input, output):
            self.gradients[name] = output[0]
        return hook

    def generate_attribution_map(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0][target_class]
        target.backward()

        attribution_maps = []
        for layer in self.target_layers:
            gradients = self.gradients[layer]
            activations = self.activations[layer]
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]
            attribution_map = torch.mean(activations, dim=1).squeeze().detach()
            attribution_maps.append(attribution_map)

        return attribution_maps

    def clean(self):
        for hook in self.hooks:
            hook.remove()
