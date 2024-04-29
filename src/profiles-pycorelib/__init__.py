from profiles_rudderstack.project import WhtProject
from .common_col_union import CommonColumnUnionModel
from .pyplot import PyPlotModel
from .attribution_model import AttributionModel

def register_extensions(project: WhtProject):
    project.register_model_type(CommonColumnUnionModel)
    project.register_model_type(AttributionModel)
    project.register_model_type(PyPlotModel)
