interface Props {
  language: 'en' | 'es';
  onChange: (lang: 'en' | 'es') => void;
}

export function LanguageSelector({ language, onChange }: Props) {
  return (
    <div className="language-selector">
      <button
        className={language === 'en' ? 'active' : ''}
        onClick={() => onChange('en')}
      >
        EN
      </button>
      <button
        className={language === 'es' ? 'active' : ''}
        onClick={() => onChange('es')}
      >
        ES
      </button>
    </div>
  );
}
