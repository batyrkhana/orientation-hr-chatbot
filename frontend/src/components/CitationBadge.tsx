export interface Citation {
  file: string;
  page: number;
}

interface Props {
  citations: Citation[];
}

export function CitationBadge({ citations }: Props) {
  if (citations.length === 0) return null;
  return (
    <div className="citations">
      {citations.map((c, i) => (
        <span key={i} className="citation-badge">
          {c.file}, p.{c.page}
        </span>
      ))}
    </div>
  );
}
