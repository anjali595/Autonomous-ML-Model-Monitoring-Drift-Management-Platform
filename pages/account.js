import React, { useEffect, useState } from 'react';
import MainLayout from '../components/MainLayout';

const AccountPage = () => {
  const [profile, setProfile] = useState(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) return;
    fetch('http://127.0.0.1:5000/auth/me', { headers: { Authorization: `Bearer ${token}` }})
      .then((r) => r.json())
      .then((data) => setProfile(data))
      .catch((e) => console.error(e));
  }, []);

  const updateProfile = () => {
    setMessage('Profile features under development');
  };

  return (
    <MainLayout>
      <div className="container">
        <h1 className="my-4">Account</h1>
        {profile ? (
          <div className="card p-3">
            <p><strong>Username:</strong> {profile.username}</p>
            <p><strong>Role:</strong> {profile.role}</p>
            <button className="btn btn-outline-primary" onClick={updateProfile}>Update profile</button>
            {message && <p className="mt-3 text-info">{message}</p>}
          </div>
        ) : (
          <p>Loading profile...</p>
        )}
      </div>
    </MainLayout>
  );
};

export default AccountPage;
