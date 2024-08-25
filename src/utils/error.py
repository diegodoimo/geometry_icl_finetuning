class DataRetrievalError(Exception):
    """Exception raised when an issue occurs while retrieving data."""
    def __init__(self, message="Requested data not found in the file system"):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
    
    
class MetricComputationError(Exception):
    """Exception raised when an issue occurs while computing the metric."""
    def __init__(self, message="An error occurred while computing the metric"):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
