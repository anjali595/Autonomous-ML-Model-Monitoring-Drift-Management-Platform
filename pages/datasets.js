import React, { useEffect, useState } from 'react';
import MainLayout from '../components/MainLayout';

const DatasetsPage = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    const load = async () => {
      const res = await fetch('http://127.0.0.1:5000/api/datasets');
      const data = await res.json();
      setDatasets(data);
      setLoading(false);
    };
    load();
  }, []);

  const handleUpload = async (e) => {
    e.preventDefault();
    await fetch('http://127.0.0.1:5000/api/datasets', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ name, description })
    });
    setName('');
    setDescription('');
    const res = await fetch('http://127.0.0.1:5000/api/datasets');
    setDatasets(await res.json());
  };

  return (
    <MainLayout>
      <div className="container">
        <h1 className="my-4">Datasets</h1>
        <form onSubmit={handleUpload} className="mb-4">
          <input className="form-control mb-2" placeholder="Dataset name" value={name} onChange={(e) => setName(e.target.value)} required />
          <textarea className="form-control mb-2" placeholder="Description" value={description} onChange={(e) => setDescription(e.target.value)} />
          <button className="btn btn-success" type="submit">Add Dataset</button>
        </form>
        {loading ? <p>Loading datasets...</p> : (
          <ul className="list-group">
            {datasets.map(ds => <li key={ds.id} className="list-group-item">{ds.name} - {ds.description}</li>)}
          </ul>
        )}
      </div>
    </MainLayout>
  );
};

export default DatasetsPage;
