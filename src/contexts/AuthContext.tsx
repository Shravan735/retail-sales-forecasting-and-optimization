import React, { createContext, useContext, useState, useEffect } from "react";

interface User {
  name: string;
  email: string;
}

interface StoredUser extends User {
  password: string;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => { success: boolean; error?: string };
  register: (name: string, email: string, password: string) => { success: boolean; error?: string };
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

const readStoredUsers = (): StoredUser[] => {
  try {
    const raw = localStorage.getItem("rso_users");
    if (!raw) return [];

    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return parsed.filter(
      (item): item is StoredUser =>
        typeof item === "object" &&
        item !== null &&
        typeof item.name === "string" &&
        typeof item.email === "string" &&
        typeof item.password === "string",
    );
  } catch {
    return [];
  }
};

const readStoredSessionUser = (): User | null => {
  try {
    const raw = localStorage.getItem("rso_user");
    if (!raw) return null;

    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      typeof parsed.name === "string" &&
      typeof parsed.email === "string"
    ) {
      return {
        name: parsed.name,
        email: parsed.email,
      };
    }
  } catch {
    return null;
  }

  return null;
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    setUser(readStoredSessionUser());
  }, []);

  const register = (name: string, email: string, password: string) => {
    const users = readStoredUsers();
    if (users.find((storedUser) => storedUser.email === email)) {
      return { success: false, error: "Email already registered" };
    }
    users.push({ name, email, password });
    localStorage.setItem("rso_users", JSON.stringify(users));
    const u = { name, email };
    localStorage.setItem("rso_user", JSON.stringify(u));
    setUser(u);
    return { success: true };
  };

  const login = (email: string, password: string) => {
    const users = readStoredUsers();
    const found = users.find((storedUser) => storedUser.email === email && storedUser.password === password);
    if (!found) return { success: false, error: "Invalid credentials" };
    const u = { name: found.name, email: found.email };
    localStorage.setItem("rso_user", JSON.stringify(u));
    setUser(u);
    return { success: true };
  };

  const logout = () => {
    localStorage.removeItem("rso_user");
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
