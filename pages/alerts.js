import React, { useEffect, useState } from 'react';
import AlertList from '../components/AlertList';
import MainLayout from '../components/MainLayout';

const AlertsPage = () => {
    const [alerts, setAlerts] = useState([]);

    useEffect(() => {
        const fetchAlerts = async () => {
            const response = await fetch('http://127.0.0.1:5000/api/alerts');
            const data = await response.json();
            setAlerts(data);
        };

        fetchAlerts();
    }, []);

    return (
        <MainLayout>
            <div className="container">
                <h1 className="my-4">Recent Alerts</h1>
                {alerts.length > 0 ? (
                    <AlertList alerts={alerts} />
                ) : (
                    <p className="text-muted">No alerts. Great! 🎉</p>
                )}
            </div>
        </MainLayout>
    );
};

export default AlertsPage;