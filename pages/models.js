import React, { useEffect, useState } from 'react';
import ModelCard from '../components/ModelCard';
import MainLayout from '../components/MainLayout';

const ModelsPage = () => {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const response = await fetch('http://127.0.0.1:5000/api/models');
                const data = await response.json();
                setModels(data);
            } catch (error) {
                console.error('Error fetching models:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchModels();
    }, []);

    return (
        <MainLayout>
            <div className="container">
                <h1 className="my-4">Models Overview</h1>
                {loading ? (
                    <p>Loading models...</p>
                ) : models.length > 0 ? (
                    <div className="row">
                        {models.map((model) => (
                            <div className="col-md-4 mb-4" key={model.id}>
                                <ModelCard model={model} />
                            </div>
                        ))}
                    </div>
                ) : (
                    <p>No models available. Please upload a model.</p>
                )}
            </div>
        </MainLayout>
    );
};

export default ModelsPage;