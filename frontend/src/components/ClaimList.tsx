import type { Claim } from "../types/chat";

interface ClaimListProps {
  claims: Claim[];
}

export default function ClaimList({ claims }: ClaimListProps) {
  if (claims.length === 0) return null;

  return (
    <ul className="claim-list">
      {claims.map((claim, i) => (
        <li className="claim" key={i}>
          <p className="claim__text">{claim.claim_text}</p>
          {claim.citation ? (
            <span className="citation-tag">
              § {claim.citation.section} · p.{claim.citation.page} · {claim.citation.chunk_id}
            </span>
          ) : (
            <span className="citation-tag citation-tag--unverified">unverified citation</span>
          )}
        </li>
      ))}
    </ul>
  );
}
