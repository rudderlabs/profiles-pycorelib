class MaterialNotFoundError(Exception):
    def __init__(self, message="this.de_ref: unable to get material"):
        self.message = message
        super().__init__(self.message)


class InvalidTransformationError(Exception):
    def __init__(self, message="transformation expression is invalid"):
        self.message = message
        super().__init__(self.message)
