import os
import torch

from model import Model


class Checkpointer():
    def __init__(self, output_dir=None, f_name='mom_sim.cpt', logger=None):
        # set output dir will this checkpoint will save itself
        self.output_dir = output_dir
        self.classifier_epoch = 0
        self.info_epoch = 0
        self.model = None
        self.optimizer = None
        self.ckp = None
        self.f_name = f_name
        self.logger = logger

    def track_new_model(self, model):
        self.model = model

    def track_new_optimizer(self, optimizer):
        self.optimizer = optimizer

    def restore_model_from_checkpoint(self, cpt_path, training_classifier=False):
        self.ckp = torch.load(cpt_path)
        hp = self.ckp['hyperparams']
        hp['logger'] = self.logger
        params = self.ckp['model']
        self.info_epoch = self.ckp['cursor']['info_epoch']
        self.classifier_epoch = self.ckp['cursor']['classifier_epoch']

        self.model = Model(**hp)

        model_dict = self.model.state_dict()
        # partial_params = {k: v for k, v in partial_params.items() if not k.startswith("project.")}
        model_dict.update(params.items())
        params = model_dict
        self.model.load_state_dict(params, strict=False)

        self.logger.info(("***** CHECKPOINTING *****\n"
                          "Model restored from checkpoint.\n"
                          "Self-supervised training epoch {}\n"
                          "Classifier training epoch {}\n"
                          "*************************"
                          .format(self.info_epoch, self.classifier_epoch)))
        return self.model

    def restore_optimizer_from_checkpoint(self, optimizer):
        params = self.ckp['optimizer']
        optimizer.load_state_dict(params)
        self.logger.info("***** CHECKPOINTING *****\n"
                         "Optimizer restored from checkpoint.\n"
                         "*************************"
                         .format(self.info_epoch, self.classifier_epoch))
        self.track_new_optimizer(optimizer)
        return optimizer

    def _get_state(self):
        return {
            'model': self.model.state_dict(),
            'hyperparams': self.model.hyperparams,
            'optimizer': self.optimizer.state_dict(),
            'cursor': {
                'info_epoch': self.info_epoch,
                'classifier_epoch': self.classifier_epoch,
            }
        }

    def _save_cpt(self):
        cpt_path = os.path.join(self.output_dir, self.f_name)
        torch.save(self._get_state(), cpt_path)
        return

    def update(self, epoch, classifier=False):
        if classifier:
            self.classifier_epoch = epoch
        else:
            self.info_epoch = epoch
        self._save_cpt()

    def get_current_position(self, classifier=False):
        if classifier:
            return self.classifier_epoch
        return self.info_epoch
