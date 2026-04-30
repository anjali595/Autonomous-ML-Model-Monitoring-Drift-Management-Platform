import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';

const LoginPage = () => {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      });

      const data = await res.json();
      if (!res.ok) {
        setError(data.message || 'Invalid credentials.');
        setLoading(false);
        return;
      }

      localStorage.setItem('token', data.token);
      router.push('/');
    } catch (err) {
      setError('Unable to login, ensure backend is running.');
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-card">
        <h2>ML Monitor Login</h2>
        <p className="text-muted">Sign in to your secure dashboard.</p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              className="input-field"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoFocus
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              className="input-field"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button disabled={loading} className="btn-primary auth-btn" type="submit">
            {loading ? 'Logging in...' : 'Sign In'}
          </button>
        </form>

        {error && <div className="alert-danger mt-2">{error}</div>}

        <p className="mt-3">
          Don’t have an account? <Link href="/signup">Create one</Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
