import { useState } from 'react';
import { v4 as uuidv4 } from 'uuid';

const SESSION_KEY = 'abc_widgets_session_id';

export function useSession(): string {
  const [sessionId] = useState<string>(() => {
    const existing = localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
    const id = uuidv4();
    localStorage.setItem(SESSION_KEY, id);
    return id;
  });
  return sessionId;
}
