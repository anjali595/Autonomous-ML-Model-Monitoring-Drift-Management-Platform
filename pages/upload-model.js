import React, { useState } from 'react';
import { useRouter } from 'next/router';
import MainLayout from '../components/MainLayout';

const UploadModelPage = () => {
  const router = useRouter();
  const [formData, setFormData] = useState({
    name: '',
    model_type: 'Custom',
    version: '1.0',
    baseline_accuracy: 0.0,
    file: null
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    if (name === 'file') {
      setFormData(prev => ({ ...prev, file: files[0] }));
    } else {
      setFormData(prev => ({ 
        ...prev, 
        [name]: name === 'baseline_accuracy' ? parseFloat(value) : value 
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError('');
    setMessage('');

    try {
      const formDataToSend = new FormData();
      formDataToSend.append('file', formData.file);
      formDataToSend.append('name', formData.name || formData.file.name.split('.')[0]);
      formDataToSend.append('model_type', formData.model_type);
      formDataToSend.append('version', formData.version);
      formDataToSend.append('baseline_accuracy', formData.baseline_accuracy);

      const response = await fetch('http://localhost:5000/api/upload-model', {
        method: 'POST',
        body: formDataToSend,
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('✅ Model uploaded successfully!');
        setFormData({
          name: '',
          model_type: 'Custom',
          version: '1.0',
          baseline_accuracy: 0.0,
          file: null
        });
        setTimeout(() => router.push('/models'), 2000);
      } else {
        setError(data.message || 'Error uploading model');
      }
    } catch (err) {
      setError('Failed to upload model: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <MainLayout>
      <div className="container" style={{ maxWidth: '600px', margin: '40px auto' }}>
        <h1 className="my-4">Upload Trained Model</h1>
        
        {message && <div style={{ color: 'green', padding: '10px', marginBottom: '10px', border: '1px solid green', borderRadius: '4px' }}>{message}</div>}
        {error && <div style={{ color: 'red', padding: '10px', marginBottom: '10px', border: '1px solid red', borderRadius: '4px' }}>{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label">Model Name (optional)</label>
            <input
              type="text"
              className="form-control"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="e.g., Fraud Detector v1"
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Model Type</label>
            <select
              className="form-control"
              name="model_type"
              value={formData.model_type}
              onChange={handleChange}
            >
              <option>Custom</option>
              <option>Random Forest</option>
              <option>Neural Network</option>
              <option>SVM</option>
              <option>XGBoost</option>
              <option>LSTM</option>
            </select>
          </div>

          <div className="mb-3">
            <label className="form-label">Version</label>
            <input
              type="text"
              className="form-control"
              name="version"
              value={formData.version}
              onChange={handleChange}
              placeholder="e.g., 1.0"
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Baseline Accuracy (0-1)</label>
            <input
              type="number"
              className="form-control"
              name="baseline_accuracy"
              value={formData.baseline_accuracy}
              onChange={handleChange}
              min="0"
              max="1"
              step="0.01"
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Select Model File</label>
            <input
              type="file"
              className="form-control"
              name="file"
              onChange={handleChange}
              accept=".pkl,.joblib,.h5,.pt,.pth,.onnx"
              required
            />
            <small className="text-muted">Supported: .pkl, .joblib, .h5, .pt, .pth, .onnx</small>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn btn-success w-100"
          >
            {loading ? 'Uploading...' : 'Upload Model'}
          </button>
        </form>
      </div>
    </MainLayout>
  );
};

export default UploadModelPage;
