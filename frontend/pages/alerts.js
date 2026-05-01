import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Clock, ShieldAlert } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import toast from 'react-hot-toast';

export default function AlertsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !user) router.push('/login');
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) fetchAlerts();
  }, [user]);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      const res = await api.get('/alerts');
      setAlerts(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      toast.error('Failed to load alerts');
    } finally {
      setLoading(false);
    }
  };

  const resolveAlert = async (id) => {
    try {
      await api.patch(`/alerts/${id}/resolve`);
      toast.success('Alert resolved');
      setAlerts(alerts.map(a => a.id === id ? { ...a, resolved: true } : a));
    } catch (err) {
      toast.error('Failed to resolve alert');
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
        </div>
      </Layout>
    );
  }

  const activeAlerts = alerts.filter(a => !a.resolved);
  const resolvedAlerts = alerts.filter(a => a.resolved);

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Alerts & Incidents</h1>
          <p className="text-muted-foreground mt-1">Manage system anomalies and data drift warnings</p>
        </div>

        {activeAlerts.length === 0 && resolvedAlerts.length === 0 ? (
          <Card className="saas-card text-center py-16">
            <ShieldAlert className="mx-auto h-12 w-12 text-muted-foreground/30 mb-4" />
            <h3 className="text-lg font-semibold text-foreground">No Alerts</h3>
            <p className="text-muted-foreground">Your monitoring systems have not raised any alerts.</p>
          </Card>
        ) : (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-rose-500" /> Active Alerts
                <Badge variant="secondary" className="ml-2">{activeAlerts.length}</Badge>
              </h3>
              
              {activeAlerts.length === 0 && (
                <div className="p-6 rounded-xl border border-dashed border-border text-center text-muted-foreground text-sm">
                  No active alerts. All clear.
                </div>
              )}

              {activeAlerts.map((alert, i) => (
                <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }} key={alert.id}>
                  <Card className="saas-card border-l-4 border-l-rose-500">
                    <CardContent className="p-5 flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <Badge className="bg-rose-500 hover:bg-rose-600 text-white shadow-sm">High Priority</Badge>
                          <span className="text-xs text-muted-foreground flex items-center">
                            <Clock className="w-3 h-3 mr-1" /> {new Date(alert.created_at).toLocaleString()}
                          </span>
                        </div>
                        <h4 className="font-semibold text-foreground mt-2">{alert.message || alert.type}</h4>
                        <p className="text-sm text-muted-foreground mt-1">Action required to resolve model degradation.</p>
                      </div>
                      <Button size="sm" onClick={() => resolveAlert(alert.id)} className="shadow-sm">
                        Resolve
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-emerald-500" /> Resolved History
              </h3>
              
              {resolvedAlerts.length === 0 && (
                <div className="p-6 rounded-xl border border-dashed border-border text-center text-muted-foreground text-sm">
                  No resolved history.
                </div>
              )}

              {resolvedAlerts.map((alert, i) => (
                <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }} key={alert.id}>
                  <Card className="saas-card bg-secondary/50 border-border opacity-75 hover:opacity-100 transition-opacity">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="text-emerald-600 border-emerald-500/30 bg-emerald-500/10">Resolved</Badge>
                        <span className="text-xs text-muted-foreground">
                          {new Date(alert.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      <h4 className="font-medium text-sm text-foreground">{alert.message || alert.type}</h4>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
