import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Upload, FileText, Loader2, ArrowLeft } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Badge } from '../components/ui/badge';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import toast from 'react-hot-toast';

export default function UploadModelPage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();
  const [formData, setFormData] = useState({
    name: '', modelType: 'Custom', version: '1.0', baselineAccuracy: 0.0, file: null,
  });
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) router.push('/login');
  }, [user, authLoading, router]);

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    if (name === 'file' && files[0]) setFormData(prev => ({ ...prev, file: files[0] }));
    else setFormData(prev => ({ ...prev, [name]: name === 'baselineAccuracy' ? parseFloat(value) : value }));
  };

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFormData(prev => ({ ...prev, file: e.dataTransfer.files[0] }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.file) return toast.error('Please select a model file');

    setLoading(true);
    try {
      const data = new FormData();
      data.append('file', formData.file);
      data.append('name', formData.name || formData.file.name.split('.')[0]);
      data.append('model_type', formData.modelType);
      data.append('version', formData.version);
      data.append('baseline_accuracy', formData.baselineAccuracy);

      await api.post('/upload-model', data, { headers: { 'Content-Type': 'multipart/form-data' } });
      toast.success('Model uploaded successfully!');
      setTimeout(() => router.push('/models'), 1500);
    } catch (err) {
      toast.error(err.response?.data?.message || 'Failed to upload model');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.back()} className="text-muted-foreground">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">Upload Model</h1>
            <p className="text-muted-foreground mt-1">Register and upload a new model for monitoring</p>
          </div>
        </div>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="saas-card">
            <CardHeader>
              <CardTitle>Model Details</CardTitle>
              <CardDescription>Provide metadata for your ML model</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid gap-5 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="name" className="text-muted-foreground font-medium">Model Name</Label>
                    <Input id="name" name="name" value={formData.name} onChange={handleChange} placeholder="e.g., Fraud Detector v1" className="h-10" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="modelType" className="text-muted-foreground font-medium">Framework / Type</Label>
                    <select
                      name="modelType" value={formData.modelType} onChange={handleChange}
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary shadow-sm"
                    >
                      <option value="Custom">Custom</option>
                      <option value="Random Forest">Random Forest</option>
                      <option value="Neural Network">Neural Network</option>
                      <option value="SVM">SVM</option>
                      <option value="XGBoost">XGBoost</option>
                    </select>
                  </div>
                </div>

                <div className="grid gap-5 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="version" className="text-muted-foreground font-medium">Version</Label>
                    <Input id="version" name="version" value={formData.version} onChange={handleChange} placeholder="1.0.0" className="h-10" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="baselineAccuracy" className="text-muted-foreground font-medium">Baseline Accuracy</Label>
                    <Input id="baselineAccuracy" name="baselineAccuracy" type="number" min="0" max="1" step="0.01" value={formData.baselineAccuracy} onChange={handleChange} placeholder="0.95" className="h-10" />
                  </div>
                </div>

                <div className="space-y-3 pt-2">
                  <Label className="text-muted-foreground font-medium">Upload File</Label>
                  <div
                    className={`border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer ${
                      dragActive ? 'border-primary bg-primary/5' : 'border-border hover:border-muted-foreground/50 hover:bg-secondary/30'
                    }`}
                    onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
                  >
                    <Upload className="mx-auto h-10 w-10 text-muted-foreground/50 mb-4" />
                    <p className="text-sm font-medium text-foreground mb-1">Drag and drop your model file here</p>
                    <p className="text-xs text-muted-foreground mb-4">Supported: .pkl, .joblib, .h5, .pt, .onnx</p>
                    
                    <input type="file" name="file" onChange={handleChange} accept=".pkl,.joblib,.h5,.pt,.pth,.onnx,.sav" className="hidden" id="file-upload" />
                    <Label htmlFor="file-upload">
                      <Button type="button" variant="secondary" asChild className="cursor-pointer">
                        <span>Browse Files</span>
                      </Button>
                    </Label>
                  </div>

                  {formData.file && (
                    <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="flex items-center space-x-3 p-4 bg-secondary rounded-lg border border-border mt-3">
                      <FileText className="h-6 w-6 text-primary" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-foreground truncate">{formData.file.name}</p>
                        <p className="text-xs text-muted-foreground">{(formData.file.size / 1024 / 1024).toFixed(2)} MB</p>
                      </div>
                      <Badge variant="outline" className="bg-background">Ready</Badge>
                    </motion.div>
                  )}
                </div>

                <div className="pt-4 border-t border-border mt-6">
                  <Button type="submit" className="w-full shadow-sm" disabled={loading || !formData.file}>
                    {loading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Uploading...</> : <><Upload className="mr-2 h-4 w-4" /> Finalize Upload</>}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </Layout>
  );
}
