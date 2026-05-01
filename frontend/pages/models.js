import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { Plus, Search, ArrowUpDown, ArrowUp, ArrowDown, Eye, Database } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import toast from 'react-hot-toast';

export default function ModelsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [models, setModels] = useState([]);
  const [filteredModels, setFilteredModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'name', direction: 'asc' });

  useEffect(() => {
    if (!authLoading && !user) router.push('/login');
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) fetchModels();
  }, [user]);

  useEffect(() => {
    filterAndSortModels();
  }, [models, searchTerm, sortConfig]);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await api.get('/models');
      setModels(Array.isArray(response.data) ? response.data : []);
    } catch (error) {
      toast.error('Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const filterAndSortModels = () => {
    let filtered = [...models];
    if (searchTerm) {
      filtered = filtered.filter(model =>
        model.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        model.model_type?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    if (sortConfig.key) {
      filtered.sort((a, b) => {
        const aValue = (a[sortConfig.key] ?? '').toString().toLowerCase();
        const bValue = (b[sortConfig.key] ?? '').toString().toLowerCase();
        if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }
    setFilteredModels(filtered);
  };

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') direction = 'desc';
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key !== key) return <ArrowUpDown className="h-4 w-4 ml-1 inline text-muted-foreground/50" />;
    return sortConfig.direction === 'asc' ? <ArrowUp className="h-4 w-4 ml-1 inline" /> : <ArrowDown className="h-4 w-4 ml-1 inline" />;
  };

  const getAccuracyColor = (accuracy) => {
    if (!accuracy) return 'text-muted-foreground';
    if (accuracy >= 0.9) return 'text-emerald-500 font-medium';
    if (accuracy >= 0.8) return 'text-amber-500 font-medium';
    return 'text-rose-500 font-medium';
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
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">Model Registry</h1>
            <p className="text-muted-foreground mt-1">Monitor and manage your deployed ML models</p>
          </div>
          <Button onClick={() => router.push('/upload-model')} className="shadow-sm">
            <Plus className="mr-2 h-4 w-4" />
            New Model
          </Button>
        </div>

        <div className="relative max-w-md">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search by name or type..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-card"
          />
        </div>

        <Card className="saas-card">
          <CardHeader>
            <CardTitle>Registered Models</CardTitle>
            <CardDescription>{filteredModels.length} model{filteredModels.length !== 1 ? 's' : ''} found</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>
                    <button onClick={() => handleSort('name')} className="font-semibold text-foreground flex items-center hover:text-primary transition-colors">
                      Name {getSortIcon('name')}
                    </button>
                  </TableHead>
                  <TableHead className="font-semibold text-foreground">Type</TableHead>
                  <TableHead>
                    <button onClick={() => handleSort('version')} className="font-semibold text-foreground flex items-center hover:text-primary transition-colors">
                      Version {getSortIcon('version')}
                    </button>
                  </TableHead>
                  <TableHead>
                    <button onClick={() => handleSort('baseline_accuracy')} className="font-semibold text-foreground flex items-center hover:text-primary transition-colors">
                      Accuracy {getSortIcon('baseline_accuracy')}
                    </button>
                  </TableHead>
                  <TableHead className="font-semibold text-foreground">Status</TableHead>
                  <TableHead className="font-semibold text-foreground">Created</TableHead>
                  <TableHead className="text-right font-semibold text-foreground">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredModels.map((model) => (
                  <TableRow key={model.id} className="hover:bg-secondary/50 transition-colors">
                    <TableCell className="font-medium text-foreground">{model.name}</TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="font-medium">{model.model_type}</Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">v{model.version}</TableCell>
                    <TableCell className={getAccuracyColor(model.baseline_accuracy)}>
                      {model.baseline_accuracy ? `${(model.baseline_accuracy * 100).toFixed(1)}%` : '—'}
                    </TableCell>
                    <TableCell>
                      {model.model_file_path ? (
                        <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Active
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span> Draft
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground text-sm">
                      {model.created_at ? new Date(model.created_at).toLocaleDateString() : '—'}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => router.push(`/predict?model=${model.id}`)}
                        className="text-muted-foreground hover:text-primary"
                        title="Test predictions"
                      >
                        <Eye className="h-4 w-4 mr-2" /> Test
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            {filteredModels.length === 0 && (
              <div className="text-center py-16">
                <Database className="mx-auto h-12 w-12 text-muted-foreground/30 mb-4" />
                <h3 className="text-lg font-semibold text-foreground mb-1">No Models Found</h3>
                <p className="text-muted-foreground mb-6">
                  {searchTerm ? 'No models match your search criteria.' : 'Get started by uploading your first model.'}
                </p>
                {!searchTerm && (
                  <Button onClick={() => router.push('/upload-model')}>
                    <Plus className="mr-2 h-4 w-4" />
                    Upload Model
                  </Button>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}