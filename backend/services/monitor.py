from datetime import datetime
import logging

class ModelMonitor:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.logger = logging.getLogger(__name__)

    def check_performance(self):
        # Implement logic to check model performance
        accuracy = self.model.evaluate(self.dataset)
        self.logger.info(f"Model {self.model.name} accuracy: {accuracy:.2f}%")
        return accuracy

    def detect_data_drift(self):
        # Implement logic to detect data drift
        drift_detected = self.dataset.check_drift()
        if drift_detected:
            self.logger.warning(f"Data drift detected for model {self.model.name}.")
        return drift_detected

    def log_monitoring_activity(self, accuracy, drift_detected):
        timestamp = datetime.now()
        log_entry = {
            'model_name': self.model.name,
            'dataset_name': self.dataset.name,
            'accuracy': accuracy,
            'data_drift_detected': drift_detected,
            'timestamp': timestamp
        }
        self.logger.info(f"Monitoring log: {log_entry}")
        # Here you would typically save the log entry to a database or file

# Example usage:
# model_monitor = ModelMonitor(model_instance, dataset_instance)
# accuracy = model_monitor.check_performance()
# drift_detected = model_monitor.detect_data_drift()
# model_monitor.log_monitoring_activity(accuracy, drift_detected)