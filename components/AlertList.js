import React from 'react';
import Link from 'next/link';

const AlertList = ({ recentAlerts = [] }) => {
    const safeAlerts = Array.isArray(recentAlerts) ? recentAlerts : [];

    return (
        <div className="alert-list">
            <h5>Recent Alerts</h5>
            {safeAlerts.length > 0 ? (
                <div className="list-group">
                    {recentAlerts.map(alert => (
                        <Link key={alert.id} href={`/alerts/${alert.id}`} className="list-group-item list-group-item-action">
                            <div className="d-flex justify-content-between align-items-start">
                                <div>
                                    <h6 className="mb-1">
                                        <span className={`badge ${alert.severity === 'high' ? 'bg-danger' : alert.severity === 'medium' ? 'bg-warning' : 'bg-info'}`}>
                                            {(alert.alert_type || '').replace('_', ' ').toUpperCase()}
                                        </span>
                                    </h6>
                                    <p className="mb-1 small">{(alert.message || '').slice(0, 60)}...</p>
                                    <small className="text-muted">{new Date(alert.created_at || alert.createdAt || Date.now()).toLocaleString()}</small>
                                </div>
                                <i className="fas fa-chevron-right"></i>
                            </div>
                        </Link>
                    ))}
                </div>
            ) : (
                <p className="text-muted text-center py-4">No alerts. Great! 🎉</p>
            )}
        </div>
    );
};

export default AlertList;