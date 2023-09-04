from profiles_rudderstack.project import WhtProject
from .model import CommonColumnUnionModel

# Registers the model type with the project
def RegisterExtensions(project: WhtProject):
    project.RegisterModelType(CommonColumnUnionModel)
