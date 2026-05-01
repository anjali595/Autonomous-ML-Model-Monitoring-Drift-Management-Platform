import React, { useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { User, LogOut, Shield, Calendar } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import { logout } from '../services/auth';

export default function AccountPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!authLoading && !user) router.push('/login');
  }, [user, authLoading, router]);

  if (authLoading) {
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
      <div className="space-y-6 max-w-xl mx-auto mt-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Account</h1>
          <p className="text-muted-foreground mt-1">Manage your profile and settings</p>
        </div>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="saas-card">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-5">
                <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-2xl font-bold shadow-sm">
                  {user?.username?.charAt(0).toUpperCase() || 'U'}
                </div>
                <div>
                  <CardTitle className="text-xl">{user?.username || 'User'}</CardTitle>
                  <p className="text-muted-foreground text-sm mt-0.5">ML Engineer</p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-4 p-4 rounded-xl border border-border bg-secondary/30">
                  <div className="p-2 bg-background rounded-lg border border-border shadow-sm">
                    <User className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-0.5">Username</p>
                    <p className="text-foreground font-medium">{user?.username || '—'}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4 p-4 rounded-xl border border-border bg-secondary/30">
                  <div className="p-2 bg-background rounded-lg border border-border shadow-sm">
                    <Shield className="w-5 h-5 text-emerald-500" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-0.5">Role</p>
                    <p className="text-foreground font-medium">Administrator</p>
                  </div>
                </div>
                <div className="flex items-center gap-4 p-4 rounded-xl border border-border bg-secondary/30">
                  <div className="p-2 bg-background rounded-lg border border-border shadow-sm">
                    <Calendar className="w-5 h-5 text-amber-500" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-0.5">Member Since</p>
                    <p className="text-foreground font-medium">2026</p>
                  </div>
                </div>

                <div className="pt-6 mt-2 border-t border-border">
                  <Button variant="destructive" onClick={logout} className="w-full shadow-sm font-semibold">
                    <LogOut className="mr-2 h-4 w-4" /> Sign Out
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </Layout>
  );
}
