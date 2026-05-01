import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { GitBranch, TrendingUp, Calendar } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import Layout from '../components/Layout';
import { motion } from 'framer-motion';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function VersionsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [modelGroups, setModelGroups] = useState({});
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !user) router.push('/login');
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) fetchVersions();
  }, [user]);

  const fetchVersions = async () => {
    try {
      setLoading(true);
      const response = await api.get('/models/versions');
      setModelGroups(response.data || {});
    } catch (error) {
      try {
        const res = await api.get('/models');
        const models = Array.isArray(res.data) ? res.data : [];
        const groups = {};
        models.forEach(m => {
          if (!groups[m.name]) groups[m.name] = [];
          groups[m.name].push(m);
        });
        setModelGroups(groups);
      } catch (e) {
        setModelGroups({});
      }
    } finally {
      setLoading(false);
    }
  };

  const modelNames = Object.keys(modelGroups);
  const selectedVersions = selectedModel ? (modelGroups[selectedModel] || []) : [];
  const chartData = selectedVersions.map(v => ({
    version: `v${v.version}`,
    accuracy: v.baseline_accuracy ? (v.baseline_accuracy * 100).toFixed(1) : 0,
  }));

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Model Versioning</h1>
          <p className="text-muted-foreground mt-1">Track model history, accuracy trends, and deployments</p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <Card className="saas-card h-full">
              <CardHeader>
                <CardTitle>Models</CardTitle>
                <CardDescription>Select a model to view versions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-1.5">
                  {modelNames.length === 0 && <p className="text-muted-foreground text-sm text-center py-4">No models found</p>}
                  {modelNames.map((name) => (
                    <motion.div
                      key={name} whileHover={{ x: 2 }}
                      className={`p-3 rounded-lg cursor-pointer transition-all border ${
                        selectedModel === name ? 'bg-primary/10 border-primary shadow-sm' : 'hover:bg-secondary border-transparent'
                      }`}
                      onClick={() => setSelectedModel(name)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-semibold text-foreground text-sm">{name}</h4>
                          <p className="text-xs text-muted-foreground mt-0.5">Latest: v{modelGroups[name][modelGroups[name].length - 1]?.version}</p>
                        </div>
                        <Badge variant="secondary" className="text-xs">{modelGroups[name].length} ver</Badge>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="lg:col-span-2">
            {selectedModel ? (
              <div className="space-y-6">
                <Card className="saas-card">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <GitBranch className="mr-2 h-5 w-5 text-primary" />
                      {selectedModel} — History
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {selectedVersions.map((ver, idx) => (
                        <motion.div key={ver.id} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }} className="flex items-center justify-between p-4 border border-border rounded-xl hover:bg-secondary/50 transition-colors">
                          <div className="flex items-center space-x-4">
                            <div className={`w-3 h-3 rounded-full ${idx === selectedVersions.length - 1 ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-muted-foreground/30'}`} />
                            <div>
                              <div className="flex items-center space-x-2">
                                <span className="font-bold text-foreground">v{ver.version}</span>
                                {idx === selectedVersions.length - 1 && <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm">Latest</Badge>}
                                <Badge variant="outline">{ver.model_type}</Badge>
                              </div>
                              <div className="flex items-center space-x-4 text-xs text-muted-foreground mt-1.5 font-medium">
                                <span className="flex items-center"><TrendingUp className="mr-1 h-3 w-3" /> {ver.baseline_accuracy ? `${(ver.baseline_accuracy * 100).toFixed(1)}%` : '—'} accuracy</span>
                                <span className="flex items-center"><Calendar className="mr-1 h-3 w-3" /> {ver.created_at ? new Date(ver.created_at).toLocaleDateString() : '—'}</span>
                              </div>
                            </div>
                          </div>
                          {ver.model_file_path && <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">Deployed</span>}
                        </motion.div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {chartData.length > 0 && (
                  <Card className="saas-card">
                    <CardHeader>
                      <CardTitle>Accuracy Trend</CardTitle>
                      <CardDescription>Performance across versions</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                          <XAxis dataKey="version" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                          <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                          <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px', color: 'hsl(var(--foreground))' }} formatter={(value) => [`${value}%`, 'Accuracy']} />
                          <Line type="monotone" dataKey="accuracy" stroke="#4f46e5" strokeWidth={3} dot={{ fill: '#4f46e5', r: 5, strokeWidth: 2, stroke: 'hsl(var(--card))' }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                )}
              </div>
            ) : (
              <Card className="saas-card h-full flex flex-col justify-center">
                <CardContent className="flex items-center justify-center py-20">
                  <div className="text-center">
                    <GitBranch className="mx-auto h-12 w-12 text-muted-foreground/30 mb-4" />
                    <h3 className="text-lg font-semibold text-foreground mb-1">Select a Model</h3>
                    <p className="text-muted-foreground text-sm">Choose a model from the list to view its version history</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}