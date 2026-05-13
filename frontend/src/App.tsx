import { useState } from 'react';
import { LanguageSelector } from './components/LanguageSelector';
import { ChatWindow } from './components/ChatWindow';
import { useSession } from './hooks/useSession';
import './App.css';

function App() {
  const [language, setLanguage] = useState<'en' | 'es'>('en');
  const sessionId = useSession();

  return (
    <div className="app">
      <header className="app-header">
        <h1>ABC Widgets HR Assistant</h1>
        <LanguageSelector language={language} onChange={setLanguage} />
      </header>
      <main className="app-main">
        <ChatWindow sessionId={sessionId} language={language} />
      </main>
    </div>
  );
}

export default App;
