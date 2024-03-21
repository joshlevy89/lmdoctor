# import extraction_utils
from .extraction_utils import Extractor
from .detection_utils import Detector
from .control_utils import Controller

class Doctor:
    def __init__(self, model, tokenizer, user_tag, assistant_tag, 
                 extraction_target=None, extraction_method=None, device='cuda:0', **kwargs):
        
        self.model = model
        self.tokenizer = tokenizer
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.device = device
        self.extractor = Extractor(model, tokenizer, user_tag, assistant_tag, device=device,
                                   extraction_target=extraction_target, extraction_method=extraction_method, 
                                   **kwargs)
        self.detector = None
        self.controller = None

    def extract(self, *args, **kwargs):
        self.extractor.extract(*args, **kwargs)
        # Initialize detector and controller once extractor has been run
        self.detector = Detector(self.extractor)
        self.controller = Controller(self.extractor)

    def __getattr__(self, name):
        for component in [self.extractor, self.detector, self.controller]:
            if component and hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(f"'Doctor' object and its components have no attribute '{name}'")
