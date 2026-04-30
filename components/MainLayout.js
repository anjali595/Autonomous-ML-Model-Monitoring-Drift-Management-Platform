import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import styles from '../styles/dashboard.module.css';

const MainLayout = ({ children }) => {
  const router = useRouter();
  const [token, setToken] = useState(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    setToken(savedToken);
    setMounted(true);

    if (!savedToken && router.pathname !== '/login' && router.pathname !== '/signup') {
      router.push('/login');
    }
  }, [router.pathname, router]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    router.push('/login');
  };

  return (
    <div className={styles.appContainer}>
      {mounted && token && (
        <nav className={styles.sidebar}>
          <div className={styles.brand}>
            <h1>ML Monitor</h1>
          </div>
          <ul>
            <li><Link href="/">Dashboard</Link></li>
            <li><Link href="/models">Models</Link></li>
            <li><Link href="/datasets">Datasets</Link></li>
            <li><Link href="/monitoring">Monitoring</Link></li>
            <li><Link href="/alerts">Alerts</Link></li>
            <li><Link href="/account">Account</Link></li>
          </ul>
          <button className={styles.rippleBtn} onClick={handleLogout}>Logout</button>
        </nav>
      )}
      <main className={styles.mainContent}>{children}</main>
    </div>
  );
};

export default MainLayout;
