import React from 'react'

interface Customer {
  id: string
  name: string
  language: string
  segment: string
  current_provider: string
  estimated_outstanding: number
  estimated_rate: number
  city: string
}

interface CustomerSelectProps {
  customers: Customer[]
  selectedCustomer: Customer | null
  onSelect: (customer: Customer) => void
}

export function CustomerSelect({ customers, selectedCustomer, onSelect }: CustomerSelectProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(amount)
  }

  const getSegmentColor = (segment: string) => {
    const colors: { [key: string]: string } = {
      high_value: '#f59e0b',
      trust_seeker: '#3b82f6',
      shakti: '#ec4899',
      young_professional: '#10b981',
    }
    return colors[segment] || '#6b7280'
  }

  return (
    <div style={styles.container}>
      {customers.map(customer => (
        <div
          key={customer.id}
          style={{
            ...styles.card,
            ...(selectedCustomer?.id === customer.id ? styles.cardSelected : {}),
          }}
          onClick={() => onSelect(customer)}
        >
          <div style={styles.header}>
            <span style={styles.name}>{customer.name}</span>
            <span
              style={{
                ...styles.segment,
                backgroundColor: getSegmentColor(customer.segment),
              }}
            >
              {customer.segment.replace('_', ' ')}
            </span>
          </div>

          <div style={styles.details}>
            <div style={styles.row}>
              <span style={styles.label}>Provider:</span>
              <span style={styles.value}>{customer.current_provider}</span>
            </div>
            <div style={styles.row}>
              <span style={styles.label}>Outstanding:</span>
              <span style={styles.value}>{formatCurrency(customer.estimated_outstanding)}</span>
            </div>
            <div style={styles.row}>
              <span style={styles.label}>Current Rate:</span>
              <span style={styles.value}>{customer.estimated_rate}%</span>
            </div>
            <div style={styles.row}>
              <span style={styles.label}>City:</span>
              <span style={styles.value}>{customer.city}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
    gap: '1rem',
  },
  card: {
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '8px',
    padding: '1rem',
    cursor: 'pointer',
    border: '2px solid transparent',
    transition: 'all 0.2s',
  },
  cardSelected: {
    borderColor: '#ED1C24',
    background: 'rgba(237,28,36,0.1)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.75rem',
  },
  name: {
    fontSize: '1rem',
    fontWeight: 'bold',
  },
  segment: {
    fontSize: '0.7rem',
    padding: '0.25rem 0.5rem',
    borderRadius: '4px',
    color: '#fff',
    textTransform: 'capitalize',
  },
  details: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
  },
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.85rem',
  },
  label: {
    opacity: 0.6,
  },
  value: {
    fontWeight: '500',
  },
}
