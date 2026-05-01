import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { FolderOpen, Plus, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import toast from 'react-hot-toast';

export default function DatasetsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) router.push('/login');
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) fetchDatasets();
  }, [user]);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const res = await api.get('/datasets');
      setDatasets(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      toast.error('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!name.trim()) return toast.error('Dataset name is required');
    setSubmitting(true);
    try {
      await api.post('/datasets', { name, description });
      setName(''); setDescription('');
      toast.success('Dataset added successfully!');
      fetchDatasets();
    } catch (err) {
      toast.error('Failed to add dataset');
    } finally {
      setSubmitting(false);
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

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Datasets</h1>
          <p className="text-muted-foreground mt-1">Manage data sources used for training and monitoring</p>
        </div>

        <Card className="saas-card">
          <CardHeader>
            <CardTitle>Add New Dataset</CardTitle>
            <CardDescription>Register a new data source</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleUpload} className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <Input placeholder="Dataset name" value={name} onChange={(e) => setName(e.target.value)} required className="h-10" />
              </div>
              <div className="flex-1">
                <Input placeholder="Description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} className="h-10" />
              </div>
              <Button type="submit" disabled={submitting} className="h-10 shadow-sm">
                {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Plus className="h-4 w-4 mr-2" />}
                Add Source
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {datasets.map((ds, i) => (
            <motion.div key={ds.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
              <Card className="saas-card hover:border-primary/50 transition-colors cursor-pointer group">
                <CardContent className="p-5">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0 group-hover:bg-primary/20 transition-colors">
                      <FolderOpen className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground text-sm leading-none mb-1.5">{ds.name}</h4>
                      <p className="text-xs text-muted-foreground line-clamp-2">{ds.description || 'No description provided.'}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {datasets.length === 0 && (
          <div className="text-center py-16">
            <FolderOpen className="mx-auto h-12 w-12 text-muted-foreground/30 mb-4" />
            <h3 className="text-lg font-semibold text-foreground">No Datasets</h3>
            <p className="text-muted-foreground">Register your first dataset to start monitoring</p>
          </div>
        )}
      </div>
    </Layout>
  );
}
