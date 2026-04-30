import React, { useEffect, useState } from 'react';
import MainLayout from '../components/MainLayout';
import Dashboard from '../components/Dashboard';

const HomePage = () => {
    const [stats, setStats] = useState({
        total_models: 0,
        total_datasets: 0,
        monitoring_logs: 0,
        active_alerts: 0,
    });
    const [recentAlerts, setRecentAlerts] = useState([]);
    const [recentLogs, setRecentLogs] = useState([]);
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000/api';

                const [statsRes, modelsRes, alertsRes, logsRes] = await Promise.all([
                    fetch(`${API_URL}/dashboard-stats`, { cache: 'no-store' }),
                    fetch(`${API_URL}/models`, { cache: 'no-store' }),
                    fetch(`${API_URL}/alerts`, { cache: 'no-store' }),
                    fetch(`${API_URL}/monitoring-logs`, { cache: 'no-store' }),
                ]);

                if (!statsRes.ok || !modelsRes.ok || !alertsRes.ok || !logsRes.ok) {
                    const errText = `API status: ${statsRes.status}, ${modelsRes.status}, ${alertsRes.status}, ${logsRes.status}`;
                    throw new Error(errText);
                }

                const statsData = await statsRes.json();
                const modelsData = await modelsRes.json();
                const alertsData = await alertsRes.json();
                const logsData = await logsRes.json();

                setStats(statsData);
                setModels(modelsData || []);
                setRecentAlerts(alertsData || []);
                setRecentLogs(logsData || []);
            } catch (err) {
                console.error('Error fetching data:', err);
                setError(`Could not connect to backend API. Confirm backend is running at ${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000/api'}. Error: ${err.message}`);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    if (loading) return <div className="container mt-5"><p>Loading dashboard...</p></div>;
    if (error) return <div className="container mt-5"><p className="text-danger">{error}</p></div>;

    return (
        <MainLayout>
            <Dashboard 
                stats={stats}
                recentAlerts={recentAlerts}
                models={models}
                recentLogs={recentLogs}
            />
        </MainLayout>
    );
};

export default HomePage;
