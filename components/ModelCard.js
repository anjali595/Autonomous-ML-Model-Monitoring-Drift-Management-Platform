import React from 'react';
import PropTypes from 'prop-types';
import styles from '../styles/dashboard.module.css';

const ModelCard = ({ model }) => {
    return (
        <div className={`card ${styles.card}`}>
            <div className="card-body">
                <h5 className="card-title">{model.name}</h5>
                <p className="card-text">
                    <span className={`badge bg-secondary`}>{model.model_type}</span>
                    <span className="ms-2 text-muted">v{model.version}</span>
                </p>
                {model.baseline_accuracy && (
                    <p className="text-muted">Baseline Accuracy: {`${(model.baseline_accuracy * 100).toFixed(2)}%`}</p>
                )}
                <a href={`/models/${model.id}`} className="btn btn-primary">View Details</a>
            </div>
        </div>
    );
};

ModelCard.propTypes = {
    model: PropTypes.shape({
        id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
        name: PropTypes.string.isRequired,
        model_type: PropTypes.string.isRequired,
        version: PropTypes.string.isRequired,
        baseline_accuracy: PropTypes.number,
    }).isRequired,
};

export default ModelCard;