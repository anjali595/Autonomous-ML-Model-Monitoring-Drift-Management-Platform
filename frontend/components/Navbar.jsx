import { motion } from 'framer-motion';
import { Bell, LogOut } from 'lucide-react';
import { useState } from 'react';
import { useRouter } from 'next/router';
import ThemeToggle from './ThemeToggle';
import { useAuth } from '../hooks/useAuth';
import { logout } from '../services/auth';
import Link from 'next/link';

export default function Navbar() {
  const { user } = useAuth();
  const router = useRouter();

  const handleLogout = () => {
    logout();
  };

  const getPageTitle = () => {
    const titles = {
      '/': 'Dashboard',
      '/models': 'Model Management',
      '/upload-model': 'Upload Model',
      '/predict': 'ML Predictions',
      '/monitoring': 'Drift Monitoring',
      '/alerts': 'Alerts',
      '/versions': 'Model Versions',
      '/datasets': 'Datasets',
      '/account': 'Account',
    };
    return titles[router.pathname] || 'AutoML Drift';
  };

  return (
    <motion.header
      initial={{ y: -30, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="sticky top-0 z-20 w-full flex items-center justify-between px-8 py-4 bg-card border-b border-border shadow-sm h-[73px]"
    >
      <div className="flex items-center gap-4">
        <h2 className="text-xl font-bold text-foreground tracking-tight">{getPageTitle()}</h2>
      </div>
      <div className="flex items-center gap-3">
        {/* Notifications */}
        <Link href="/alerts" passHref legacyBehavior>
          <motion.a
            whileTap={{ scale: 0.85 }}
            className="relative p-2 rounded-lg text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors cursor-pointer"
            aria-label="Notifications"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 bg-destructive text-destructive-foreground rounded-full w-2 h-2 flex items-center justify-center font-bold animate-pulse ring-2 ring-card" />
          </motion.a>
        </Link>

        {/* Theme Toggle */}
        <ThemeToggle />

        {/* User info */}
        <div className="flex items-center gap-3 pl-4 border-l border-border ml-1">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-sm font-bold shadow-sm">
            {user?.username?.charAt(0).toUpperCase() || 'U'}
          </div>
          <span className="text-sm text-foreground font-medium hidden sm:block">
            {user?.username || 'User'}
          </span>
          <motion.button
            whileTap={{ scale: 0.85 }}
            onClick={handleLogout}
            className="p-2 rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors group ml-1"
            aria-label="Logout"
            title="Logout"
          >
            <LogOut className="w-4 h-4 transition-colors" />
          </motion.button>
        </div>
      </div>
    </motion.header>
  );
}
