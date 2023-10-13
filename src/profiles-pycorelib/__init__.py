from profiles_rudderstack.project import WhtProject
from .common_col_union import CommonColumnUnionModel

def register_extensions(project: WhtProject):
    project.register_model_type(CommonColumnUnionModel)
