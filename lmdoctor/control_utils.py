from .utils import format_prompt
from .detection_utils import generate_and_project

class Controller:
    """
    Wrapper around model that enables it to control generation by manipulating representation of
    concepts at inference time. 
    """
    def __init__(self, extractor, transformer_layers=None):
        self.model = extractor.model
        self.tokenizer = extractor.tokenizer
        self.user_tag = extractor.user_tag
        self.assistant_tag = extractor.assistant_tag
        self.device = extractor.device
        self.direction_info = extractor.direction_info
        self.directions = extractor.direction_info['directions']
        if transformer_layers is None:
            self.transformer_layers = self.get_transformer_layers()

    def _clear_hooks(self, model):
        for module in self.transformer_layers:
            module._forward_hooks.clear()
    
    def generate_with_control(self, prompt, control_direction=None, alpha=1, 
                              n_trim_layers=10, control_layers=None,
                              control_gen=True, control_prompt=False, 
                              gen_only=True, return_projections=True, should_format_prompt=True,
                              **kwargs):
        """
        Adds/subtracts representation of a concept at inference time. 
        control_direction: None means no control (vanilla generation); 1 adds the vector; -1 subtracts it; 
        alpha controls the gain of manipulation (must be > 0): 
            - Setting alpha to 1 uses the direction vector directly. Values below or above 1 scale it down/up. 
            - If responses are incoherent, you can reduce alpha. If the manipulation is not strong enough, increase alpha.
        n_trim_layers: number of layers to NOT manipulate on either side of model. 0 would manipulate all layers.
        control_gen/control_prompt: applies control element to generated text and input prompt, respectively. if both
        set to False, no control is applied.
        """

        if alpha <= 0:
            raise ValueError(f"alpha must be >0, but is {alpha}")

        if control_direction not in [None, -1, 1]:
            raise ValueError(f"contol_direction is {control_direction} but must be "\
                             "None (no control), 1 (adds vector) or -1 (subtracts vector)")
        
        if control_direction:
            # add hooks 
            self._clear_hooks(self.model) # good practice to clear hooks first
            
            if control_layers and n_trim_layers:
                raise ValueError('Cannot specify both control_layers and n_trim_layers')
            elif control_layers:
                layers = control_layers 
            elif n_trim_layers:
                start_layer, end_layer = n_trim_layers, self.model.config.num_hidden_layers-n_trim_layers
                layers = range(start_layer, end_layer)
            else:
                raise ValueError('Must specify either control_layers or n_trim_layers')
                
            for layer in layers:
                def hook(m, inp, op):
                    if op[0].shape[1] > 1: # corresponds to the prompt, which is passed as a chunk
                        if control_prompt:
                            op[0][0, :, :] += alpha * self.directions[layer] / self.directions[layer].norm()  * control_direction
                        else:
                            return op
                    else: # corresponds to the text generation, which is passed one at a time
                        if control_gen:
                            op[0][0, 0, :] += alpha * self.directions[layer] / self.directions[layer].norm()  * control_direction
                        else:
                            return op
                    return op
                # per https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L710, 
                # the first value in module output (used in hook) is the input to the layer
                h = self.transformer_layers[layer].register_forward_hook(hook)
            
        output = generate_and_project(self, prompt, direction_info=self.direction_info, 
                                 gen_only=gen_only, return_projections=return_projections, should_format_prompt=should_format_prompt,
                                 **kwargs)

        # clear hooks
        self._clear_hooks(self.model)
        return output

    
    def get_transformer_layers(self):
        try:
            return self.model.model.layers # mistral and llama
        except:
            try:
                return self.model.transformer.h # gpt-2
            except:
                raise ValueError('Could not locate the transformer layers module list within model. Can pass the module' \
                                 'directly with transformer_layers when instantiating controller')

    def __getattr__(self, name):
        return getattr(self.model, name)
        
    