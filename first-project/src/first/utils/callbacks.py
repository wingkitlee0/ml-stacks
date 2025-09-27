from ray.train.lightning import RayTrainReportCallback


class MyRayTrainReportCallback(RayTrainReportCallback):
    def on_train_epoch_end(self, trainer, pl_module):
        print("MyRayTrainReportCallback")
        super().on_train_epoch_end(trainer, pl_module)
