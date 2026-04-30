import React from 'react';
import styles from '../styles/dashboard.module.css';
import ModelCard from './ModelCard';
import AlertList from './AlertList';

const Dashboard = ({ stats = {}, recentAlerts = [], models = [], recentLogs = [] }) => {
    const safeModels = Array.isArray(models) ? models : [];
    const safeAlerts = Array.isArray(recentAlerts) ? recentAlerts : [];
    const safeLogs = Array.isArray(recentLogs) ? recentLogs : [];

    return (
        <div className="container-fluid">
            <h2 className="mb-4"><i className="fas fa-tachometer-alt"></i> Dashboard</h2>

            {/* Statistics Cards */}
            <div className="row mb-4">
                <div className="col-md-3 mb-3">
                    <div className={`card text-white bg-primary ${styles.card}`}>
                        <div className="card-body">
                            <div className="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 className="card-title mb-0">Total Models</h6>
                                    <h3 className="mb-0">{stats.total_models}</h3>
                                </div>
                                <i className="fas fa-robot fa-3x opacity-50"></i>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-md-3 mb-3">
                    <div className={`card text-white bg-success ${styles.card}`}>
                        <div className="card-body">
                            <div className="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 className="card-title mb-0">Total Datasets</h6>
                                    <h3 className="mb-0">{stats.total_datasets}</h3>
                                </div>
                                <i className="fas fa-database fa-3x opacity-50"></i>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-md-3 mb-3">
                    <div className={`card text-white bg-warning ${styles.card}`}>
                        <div className="card-body">
                            <div className="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 className="card-title mb-0">Monitoring Logs</h6>
                                    <h3 className="mb-0">{stats.recent_logs}</h3>
                                </div>
                                <i className="fas fa-chart-line fa-3x opacity-50"></i>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-md-3 mb-3">
                    <div className={`card text-white bg-danger ${styles.card}`}>
                        <div className="card-body">
                            <div className="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 className="card-title mb-0">Active Alerts</h6>
                                    <h3 className="mb-0">{stats.active_alerts}</h3>
                                </div>
                                <i className="fas fa-bell fa-3x opacity-50"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="row">
                {/* Recent Alerts */}
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header bg-light d-flex justify-content-between align-items-center">
                            <h5 className="mb-0">Recent Alerts</h5>
                            <a href="/alerts" className="btn btn-sm btn-outline-primary">View All</a>
                        </div>
                        <div className="card-body">
                            <AlertList alerts={recentAlerts} />
                        </div>
                    </div>
                </div>

                {/* Models Overview */}
                <div className="col-lg-6 mb-4">
                    <div className="card">
                        <div className="card-header bg-light d-flex justify-content-between align-items-center">
                            <h5 className="mb-0">Models Overview</h5>
                            <a href="/models" className="btn btn-sm btn-outline-primary">+ Upload</a>
                        </div>
                        <div className="card-body">
                            {safeModels.length > 0 ? (
                                safeModels.map(model => (
                                    <ModelCard key={model.id} model={model} />
                                ))
                            ) : (
                                <p className="text-muted text-center py-4">No models yet. <a href="/models">Upload one</a></p>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Monitoring Logs */}
            <div className="row">
                <div className="col-12">
                    <div className="card">
                        <div className="card-header bg-light">
                            <h5 className="mb-0">Recent Monitoring Activity</h5>
                        </div>
                        <div className="card-body">
                            {(safeLogs.length > 0) ? (
                                <table className="table table-hover">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Dataset</th>
                                            <th>Accuracy</th>
                                            <th>Data Drift</th>
                                            <th>Model Drift</th>
                                            <th>PSI Score</th>
                                            <th>Timestamp</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {safeLogs.map(log => (
                                            <tr key={log.id}>
                                                <td>
                                                    {log.model_id || 'N/A'}
                                                </td>
                                                <td>
                                                    {log.dataset_id ? log.dataset_id : <span className="text-muted">N/A</span>}
                                                </td>
                                                <td>
                                                    {log.accuracy ? (
                                                        <span className="badge bg-success">{(log.accuracy * 100).toFixed(2)}%</span>
                                                    ) : (
                                                        <span className="text-muted">N/A</span>
                                                    )}
                                                </td>
                                                <td>
                                                    <span className={`badge ${log.data_drift_detected ? 'bg-warning' : 'bg-success'}`}>
                                                        {log.data_drift_detected ? 'Detected' : 'None'}
                                                    </span>
                                                </td>
                                                <td>
                                                    <span className={`badge ${log.model_drift_detected ? 'bg-danger' : 'bg-success'}`}>
                                                        {log.model_drift_detected ? 'Detected' : 'None'}
                                                    </span>
                                                </td>
                                                <td>
                                                    {log.psi_score ? log.psi_score.toFixed(4) : <span className="text-muted">N/A</span>}
                                                </td>
                                                <td>
                                                    <small className="text-muted">{new Date(log.timestamp).toLocaleString()}</small>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (
                                <p className="text-muted text-center py-4">No monitoring logs yet. Upload a model and dataset to get started!</p>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;