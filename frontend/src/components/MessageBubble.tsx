import { CitationBadge, Citation } from './CitationBadge';

interface Props {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

export function MessageBubble({ role, content, citations = [] }: Props) {
  return (
    <div className={`message-bubble ${role}`}>
      <p>{content}</p>
      {role === 'assistant' && <CitationBadge citations={citations} />}
    </div>
  );
}
