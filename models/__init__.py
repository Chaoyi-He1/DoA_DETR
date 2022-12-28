from .detr import build


def build_model(args, hyp):
    return build(args, hyp)
