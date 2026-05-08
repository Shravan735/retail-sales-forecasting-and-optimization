import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { BarChart3, TrendingUp, Brain } from "lucide-react";

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [welcomeUser, setWelcomeUser] = useState("");
  const { login, register, user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (user && !welcomeUser) {
      navigate("/upload");
    }
  }, [user, welcomeUser, navigate]);

  if (user && !welcomeUser) {
    return null;
  }

  const readSessionUserName = () => {
    try {
      const raw = localStorage.getItem("rso_user");
      if (!raw) return "";

      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null && typeof parsed.name === "string") {
        return parsed.name;
      }
    } catch {
      return "";
    }

    return "";
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (isLogin) {
      const result = login(email, password);
      if (result.success) {
        setWelcomeUser(readSessionUserName());
      } else {
        setError(result.error || "Login failed");
      }
    } else {
      if (!name.trim()) { setError("Name is required"); return; }
      const result = register(name, email, password);
      if (result.success) {
        setWelcomeUser(name);
      } else {
        setError(result.error || "Registration failed");
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden">
      {/* Subtle grid background */}
      <div className="absolute inset-0 opacity-5" style={{
        backgroundImage: "linear-gradient(hsl(var(--primary) / 0.3) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--primary) / 0.3) 1px, transparent 1px)",
        backgroundSize: "60px 60px"
      }} />

      <AnimatePresence mode="wait">
        {welcomeUser ? (
          <motion.div
            key="welcome"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.1 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-center z-10"
            onAnimationComplete={() => {
              setTimeout(() => navigate("/upload"), 2000);
            }}
          >
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
            >
              <Brain className="w-16 h-16 text-primary mx-auto mb-6" />
              <h1 className="text-4xl font-bold text-foreground mb-3">
                Welcome, <span className="text-primary">{welcomeUser}</span>
              </h1>
              <p className="text-muted-foreground text-lg">Initializing forecasting engine...</p>
            </motion.div>

            {/* Animated dots */}
            <div className="flex gap-2 justify-center mt-8">
              {[0, 1, 2].map(i => (
                <motion.div
                  key={i}
                  className="w-2 h-2 rounded-full bg-primary"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                />
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.5 }}
            className="w-full max-w-md z-10 px-6"
          >
            {/* Header */}
            <div className="text-center mb-8">
              <div className="flex items-center justify-center gap-3 mb-4">
                <BarChart3 className="w-8 h-8 text-primary" />
                <TrendingUp className="w-6 h-6 text-primary/60" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">RSO Forecaster</h1>
              <p className="text-muted-foreground text-sm mt-1">
                Retail Sales Forecasting & Optimization
              </p>
            </div>

            {/* Card */}
            <div className="rounded-xl border border-border bg-card p-8 shadow-2xl shadow-primary/5">
              {/* Toggle */}
              <div className="flex rounded-lg bg-muted p-1 mb-6">
                {["Login", "Register"].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => { setIsLogin(tab === "Login"); setError(""); }}
                    className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${
                      (tab === "Login" ? isLogin : !isLogin)
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <AnimatePresence mode="wait">
                  {!isLogin && (
                    <motion.div
                      key="name"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Label htmlFor="name" className="text-foreground/80">Full Name</Label>
                      <Input
                        id="name"
                        value={name}
                        onChange={e => setName(e.target.value)}
                        placeholder="John Doe"
                        className="mt-1 bg-muted border-border focus:border-primary"
                      />
                    </motion.div>
                  )}
                </AnimatePresence>

                <div>
                  <Label htmlFor="email" className="text-foreground/80">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                    className="mt-1 bg-muted border-border focus:border-primary"
                  />
                </div>

                <div>
                  <Label htmlFor="password" className="text-foreground/80">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="********"
                    required
                    minLength={6}
                    className="mt-1 bg-muted border-border focus:border-primary"
                  />
                </div>

                {error && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-destructive text-sm"
                  >
                    {error}
                  </motion.p>
                )}

                <Button type="submit" className="w-full font-semibold">
                  {isLogin ? "Sign In" : "Create Account"}
                </Button>
              </form>
            </div>

            <p className="text-center text-muted-foreground text-xs mt-6">
              ML-powered forecasting using LSTM, XGBoost & Random Forest
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Auth;
