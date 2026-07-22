# Security Policy

## Reporting A Vulnerability

The project intends to receive vulnerability reports through GitHub's private
vulnerability reporting flow:

<https://github.com/andreaborio/hebrus/security/advisories/new>

Private vulnerability reporting is not enabled for this repository yet.
Enabling it is a required administrative step before public launch; until then,
the advisory URL above will not accept a report. A reporter may open a minimal
public issue requesting private security contact, but that issue must contain
no vulnerability details, reproduction steps, affected versions, logs, or
attachments.

Never disclose an undisclosed vulnerability in a public issue, discussion, or
pull request. Public details can expose users before a fix or mitigation is
available.

A useful private report includes:

- the affected commit, tag, executable, and runtime configuration;
- a concise description of the impact and the conditions required to trigger
  it;
- reproduction steps or a minimal proof of concept, with secrets and model
  data removed;
- any known workaround or mitigation; and
- whether the issue has already been disclosed elsewhere.

Once private reporting is enabled, the maintainers will use the advisory to
investigate, coordinate a fix, and agree on disclosure details with the
reporter. This project does not promise a fixed acknowledgement or remediation
time. Complexity, hardware access, model availability, and impact can all
affect the investigation.

## Supported Versions

The project does not currently publish a version-by-version security support
matrix. Include the exact Git commit or tag in every report. Historical tags,
benchmark records, and model artifacts are not by themselves a promise of
ongoing security support.

## Scope And Disclosure

Security reports may cover the engine, command-line tools, HTTP server, agent,
model conversion and download tooling, file-format parsing, or build and
release infrastructure in this repository. Model quality problems without a
security impact belong in the normal issue tracker.

Please keep vulnerability details in the private advisory until the
maintainers and reporter have coordinated a public disclosure. Never attach
private model weights, API keys, access tokens, personal data, or unrelated
system contents to a report.
