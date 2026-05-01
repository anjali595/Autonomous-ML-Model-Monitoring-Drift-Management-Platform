import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Brain, Play, Loader2, AlertCircle, Info } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Badge } from '../components/ui/badge';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import toast from 'react-hot-toast';

const LOAN_FEATURE_NAMES = [
  'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
  'Loan_Amount_Term', 'Credit_History', 'Property_Area'
];

const LOAN_DEFAULTS = [1, 1, 0, 1, 0, 5000, 2000, 150, 360, 1, 2];

const PredictPage = () => {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [features, setFeatures] = useState(LOAN_DEFAULTS.map(String));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) fetchModels();
  }, [user]);

  useEffect(() => {
    if (router.query.model && models.length > 0) {
      setSelectedModelId(router.query.model);
    }
  }, [router.query.model, models]);

  const fetchModels = async () => {
    try {
      const res = await api.get('/models');
      const data = Array.isArray(res.data) ? res.data : [];
      const uploadedModels = data.filter(m => m.model_file_path);
      setModels(uploadedModels);
      if (uploadedModels.length > 0 && !selectedModelId) {
        setSelectedModelId(String(uploadedModels[0].id));
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    } finally {
      setModelsLoading(false);
    }
  };

  const handleFeatureChange = (index, value) => {
    const updated = [...features];
    updated[index] = value;
    setFeatures(updated);
  };

  const handlePredict = async () => {
    if (!selectedModelId) {
      toast.error('Please select a model');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const featureValues = features.map(f => parseFloat(f) || 0);
      const response = await api.post(`/predict/${selectedModelId}`, {
        features: featureValues,
      });
      setResult(response.data);
      toast.success('Prediction complete!');
    } catch (err) {
      const msg = err.response?.data?.message || 'Prediction failed';
      toast.error(msg);
      setResult({ error: msg });
    } finally {
      setLoading(false);
    }
  };

  const loadDefaults = () => {
    setFeatures(LOAN_DEFAULTS.map(String));
    setResult(null);
  };

  const selectedModel = models.find(m => String(m.id) === String(selectedModelId));

  if (modelsLoading) {
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
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">ML Predictions</h1>
          <p className="text-muted-foreground mt-1">Test your models with real-time predictions</p>
        </div>

        <div className="grid gap-6 lg:grid-cols-5">
          <div className="lg:col-span-3 space-y-6">
            <Card className="saas-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary" />
                  Select Model
                </CardTitle>
              </CardHeader>
              <CardContent>
                {models.length === 0 ? (
                  <div className="text-center py-6">
                    <p className="text-muted-foreground mb-3 text-sm">No deployed models found.</p>
                    <Button variant="outline" onClick={() => router.push('/upload-model')}>Upload a Model</Button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {models.map(m => (
                      <motion.div
                        key={m.id}
                        whileHover={{ x: 2 }}
                        onClick={() => { setSelectedModelId(String(m.id)); setResult(null); }}
                        className={`p-3 rounded-lg cursor-pointer transition-all border ${
                          String(m.id) === String(selectedModelId)
                            ? 'bg-primary/10 border-primary shadow-sm'
                            : 'hover:bg-secondary border-border'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="text-sm font-semibold text-foreground">{m.name}</h4>
                            <p className="text-xs text-muted-foreground mt-0.5">{m.model_type} • v{m.version}</p>
                          </div>
                          <Badge variant="secondary" className="text-xs">
                            {m.baseline_accuracy ? `${(m.baseline_accuracy * 100).toFixed(0)}%` : '—'}
                          </Badge>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="saas-card">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Input Features</CardTitle>
                    <CardDescription>
                      {selectedModel ? `Features for ${selectedModel.name}` : 'Enter feature values'}
                    </CardDescription>
                  </div>
                  <Button variant="outline" size="sm" onClick={loadDefaults}>
                    Reset
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-2">
                  {features.map((val, idx) => (
                    <div key={idx} className="space-y-1.5">
                      <Label className="text-xs font-medium text-muted-foreground">
                        {LOAN_FEATURE_NAMES[idx] || `Feature ${idx + 1}`}
                      </Label>
                      <Input
                        type="number"
                        value={val}
                        onChange={(e) => handleFeatureChange(idx, e.target.value)}
                        className="h-9 text-sm"
                      />
                    </div>
                  ))}
                </div>

                <div className="mt-5 p-4 rounded-lg bg-secondary border border-border">
                  <p className="text-xs text-muted-foreground flex items-start gap-2">
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5 text-primary" />
                    <span><strong>Guide:</strong> Gender (1=M, 0=F), Married (1=Y, 0=N), Education (1=Grad, 0=Not), Credit_History (1=Y, 0=N), Property_Area (0=Rural, 1=Semi, 2=Urban)</span>
                  </p>
                </div>

                <Button
                  className="w-full mt-6 shadow-sm"
                  onClick={handlePredict}
                  disabled={loading || !selectedModelId}
                >
                  {loading ? (
                    <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Running Inference...</>
                  ) : (
                    <><Play className="mr-2 h-4 w-4 fill-current" />Run Prediction</>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          <div className="lg:col-span-2">
            <Card className="saas-card sticky top-24">
              <CardHeader>
                <CardTitle>Result</CardTitle>
                <CardDescription>Inference output and confidences</CardDescription>
              </CardHeader>
              <CardContent>
                {!result ? (
                  <div className="text-center py-16">
                    <Brain className="mx-auto h-12 w-12 text-muted-foreground/30 mb-4" />
                    <p className="text-muted-foreground text-sm">Waiting for prediction run</p>
                  </div>
                ) : result.error ? (
                  <div className="text-center py-8">
                    <AlertCircle className="mx-auto h-10 w-10 text-rose-500 mb-3" />
                    <p className="text-rose-500 text-sm font-medium">{result.error}</p>
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-6"
                  >
                    <div className={`text-center p-6 rounded-xl border ${result.prediction === 1 ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-rose-500/10 border-rose-500/20'}`}>
                      <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Model Output</p>
                      <div className="text-4xl mb-4">
                        {result.prediction === 1 ? '✅' : '❌'}
                      </div>
                      <Badge className={`text-sm px-3 py-1 ${result.prediction === 1 ? 'bg-emerald-500 text-white hover:bg-emerald-600' : 'bg-rose-500 text-white hover:bg-rose-600'}`}>
                        {result.prediction === 1 ? 'Approved' : 'Rejected'}
                      </Badge>
                    </div>

                    {result.probabilities && (
                      <div className="space-y-4 pt-2">
                        <p className="text-sm font-semibold text-foreground">Confidence Scores</p>
                        {result.probabilities.map((prob, idx) => (
                          <div key={idx} className="space-y-1.5">
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground font-medium">
                                {idx === 0 ? 'Rejected' : 'Approved'}
                              </span>
                              <span className="text-foreground font-bold">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-2 rounded-full bg-secondary overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${prob * 100}%` }}
                                transition={{ duration: 0.8, delay: 0.2 }}
                                className={`h-full rounded-full ${idx === 1 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default PredictPage;
