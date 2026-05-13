import { useState, useRef, useEffect } from 'react';
import { MessageBubble } from './MessageBubble';
import { Citation } from './CitationBadge';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

interface Props {
  sessionId: string;
  language: 'en' | 'es';
}

export function ChatWindow({ sessionId, language }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text, language }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: language === 'es'
          ? 'Error al conectar con el servidor. Intenta de nuevo.'
          : 'Error connecting to server. Please try again.',
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.map((msg, i) => (
          <MessageBubble key={i} role={msg.role} content={msg.content} citations={msg.citations} />
        ))}
        {loading && <div className="loading-indicator">…</div>}
        <div ref={bottomRef} />
      </div>
      <div className="input-row">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={language === 'es' ? 'Escribe tu pregunta…' : 'Type your question…'}
          rows={2}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {language === 'es' ? 'Enviar' : 'Send'}
        </button>
      </div>
    </div>
  );
}
