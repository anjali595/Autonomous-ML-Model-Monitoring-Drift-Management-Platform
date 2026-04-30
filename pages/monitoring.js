import React, { useEffect, useState } from 'react';
import MainLayout from '../components/MainLayout';

const MonitoringPage = () => {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    const load = async () => {
      const res = await fetch('http://127.0.0.1:5000/api/monitoring-logs');
      const data = await res.json();
      setLogs(data);
    };
    load();
  }, []);

  return (
    <MainLayout>
      <div className="container">
        <h1 className="my-4">Monitoring Logs</h1>
        <table className="table table-striped">
          <thead>
            <tr>
              <th>ID</th><th>Model</th><th>Dataset</th><th>Accuracy</th><th>Data Drift</th><th>Model Drift</th><th>Time</th>
            </tr>
          </thead>
          <tbody>
            {logs.map(log => (
              <tr key={log.id}>
                <td>{log.id}</td>
                <td>{log.model_id}</td>
                <td>{log.dataset_id}</td>
                <td>{log.accuracy ? `${(log.accuracy*100).toFixed(2)}%` : 'N/A'}</td>
                <td>{log.data_drift_detected ? 'Yes' : 'No'}</td>
                <td>{log.model_drift_detected ? 'Yes' : 'No'}</td>
                <td>{new Date(log.timestamp).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </MainLayout>
  );
};

export default MonitoringPage;
