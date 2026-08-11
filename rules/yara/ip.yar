rule ContainsIPAddress
{
    meta:
        category = "Sensitive Data"
        description = "Detects IPv4 and IPv6 addresses in a prompt, which may indicate exfiltration targets, C2 endpoints, or internal network disclosure"

    strings:
        $ipv4 = /\b(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\b/
        $ipv6 = /\b([0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b/

    condition:
        any of them
}
